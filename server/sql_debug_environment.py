"""
SQL Debug Environment Implementation.

An AI agent is given a broken SQL query and a simulated SQLite database.
It must use tools to diagnose and fix the query.

Three tasks with increasing difficulty:
  - easy:   Single syntax error in a simple SELECT query
  - medium: Wrong column name in a JOIN across two tables
  - hard:   Multiple issues — wrong aggregation, bad filter, missing GROUP BY

Reward shaping (partial credit at every step):
  +0.10  valid tool call in a useful order
  +0.20  inspect_schema before run_query (good diagnosis pattern)
  +0.30  run_query returns rows (query is at least syntactically valid)
  +0.50  validate_fix called with a query that runs without error
  +1.00  validate_fix called with a query that returns exactly the expected output
  -0.05  invalid tool call or unknown tool
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SQLDebugAction, SQLDebugObservation
except ImportError:
    from sql_fixit_rl_agent.models import SQLDebugAction, SQLDebugObservation


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {

    # ------------------------------------------------------------------
    # EASY: Single syntax error — missing FROM keyword
    # ------------------------------------------------------------------
    "easy": {
        "description": (
            "Fix the broken SQL query. The query is supposed to return all "
            "customers who placed an order in 2024. "
            "Find the syntax error and provide the corrected SQL."
        ),
        "broken_sql": "SELECT name, email CUSTOMERS WHERE order_year = 2024;",
        "setup_sql": [
            """
            CREATE TABLE customers (
                id       INTEGER PRIMARY KEY,
                name     TEXT,
                email    TEXT,
                order_year INTEGER
            )
            """,
            "INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com', 2024)",
            "INSERT INTO customers VALUES (2, 'Bob',   'bob@example.com',   2023)",
            "INSERT INTO customers VALUES (3, 'Carol', 'carol@example.com', 2024)",
        ],
        # Canonical correct answer — what the fixed query must return
        "expected_sql": "SELECT name, email FROM customers WHERE order_year = 2024;",
        "expected_rows": [
            ("Alice", "alice@example.com"),
            ("Carol", "carol@example.com"),
        ],
        "max_steps": 6,
    },

    # ------------------------------------------------------------------
    # MEDIUM: Wrong column name in a JOIN
    # ------------------------------------------------------------------
    "medium": {
        "description": (
            "Fix the broken SQL query. The query is supposed to return each "
            "order's ID, the customer name, and the order total. "
            "There is a wrong column reference in the JOIN condition. "
            "Inspect the schema carefully, then fix and validate."
        ),
        "broken_sql": (
            "SELECT orders.id, customers.name, orders.total "
            "FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id "
            "WHERE orders.total > 100;"
        ),
        "setup_sql": [
            """
            CREATE TABLE customers (
                id     INTEGER PRIMARY KEY,
                name   TEXT,
                email  TEXT
            )
            """,
            """
            CREATE TABLE orders (
                id          INTEGER PRIMARY KEY,
                customer_id INTEGER,
                total       REAL
            )
            """,
            "INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com')",
            "INSERT INTO customers VALUES (2, 'Bob',   'bob@example.com')",
            "INSERT INTO orders VALUES (101, 1, 250.0)",
            "INSERT INTO orders VALUES (102, 2,  80.0)",
            "INSERT INTO orders VALUES (103, 1, 175.0)",
        ],
        "expected_sql": (
            "SELECT orders.id, customers.name, orders.total "
            "FROM orders "
            "JOIN customers ON orders.customer_id = customers.id "
            "WHERE orders.total > 100;"
        ),
        "expected_rows": [
            (101, "Alice", 250.0),
            (103, "Alice", 175.0),
        ],
        "max_steps": 10,
    },

    # ------------------------------------------------------------------
    # HARD: Multiple issues — wrong aggregate + missing GROUP BY + wrong filter
    # ------------------------------------------------------------------
    "hard": {
        "description": (
            "Fix the broken SQL query. The query is supposed to return each "
            "product category along with its total sales revenue, but only "
            "for categories with total revenue above 500. "
            "There are THREE bugs: (1) wrong aggregate function used, "
            "(2) missing GROUP BY clause, (3) WHERE used instead of HAVING "
            "for the aggregate filter. Fix all three."
        ),
        "broken_sql": (
            "SELECT category, AVG(price * quantity) AS revenue "
            "FROM sales "
            "WHERE (price * quantity) > 500 "
            "ORDER BY revenue DESC;"
        ),
        "setup_sql": [
            """
            CREATE TABLE sales (
                id       INTEGER PRIMARY KEY,
                category TEXT,
                price    REAL,
                quantity INTEGER
            )
            """,
            "INSERT INTO sales VALUES (1, 'Electronics', 200.0, 3)",
            "INSERT INTO sales VALUES (2, 'Electronics', 150.0, 2)",
            "INSERT INTO sales VALUES (3, 'Clothing',     50.0, 5)",
            "INSERT INTO sales VALUES (4, 'Clothing',     80.0, 4)",
            "INSERT INTO sales VALUES (5, 'Books',        20.0, 10)",
            "INSERT INTO sales VALUES (6, 'Books',        15.0,  3)",
        ],
        "expected_sql": (
            "SELECT category, SUM(price * quantity) AS revenue "
            "FROM sales "
            "GROUP BY category "
            "HAVING SUM(price * quantity) > 500 "
            "ORDER BY revenue DESC;"
        ),
        "expected_rows": [
            ("Electronics", 900.0),
            ("Clothing",    570.0),
        ],
        "max_steps": 15,
    },
}

VALID_TOOLS = {"list_tables", "inspect_schema", "run_query", "validate_fix"}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLDebugEnvironment(Environment):
    """
    SQL Debug Environment.

    The agent receives a broken SQL query and a simulated SQLite database.
    It must call tools to diagnose and fix the query within a step budget.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name!r}. Choose from {list(TASKS)}")

        self._task_name = task_name
        self._task = TASKS[task_name]
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._conn: Optional[sqlite3.Connection] = None
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._schema_inspected: bool = False   # reward bonus tracking
        self._valid_run_attempted: bool = False
        self._solved: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> SQLDebugObservation:
        """Reset the environment and set up the database.
        
        Args:
            **kwargs: Optional parameters. Supports 'task' to switch task dynamically.
        """
        # Allow task switching via reset(task="medium")
        task_name = kwargs.get("task", self._task_name)
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name!r}. Choose from {list(TASKS)}")
        self._task_name = task_name
        self._task = TASKS[task_name]

        # Close previous connection if any
        if self._conn:
            self._conn.close()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_reward = 0.0
        self._done = False
        self._schema_inspected = False
        self._valid_run_attempted = False
        self._solved = False

        # Create fresh in-memory SQLite database
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        for sql in self._task["setup_sql"]:
            self._conn.execute(sql)
        self._conn.commit()

        tables = self._get_table_names()

        return SQLDebugObservation(
            tool_result="Environment ready. Use list_tables to explore the database.",
            tool_valid=True,
            task_description=self._task["description"],
            broken_sql=self._task["broken_sql"],
            step=0,
            cumulative_reward=0.0,
            available_tables=tables,
            done=False,
            reward=0.0,
        )

    def step(self, action: SQLDebugAction) -> SQLDebugObservation:  # type: ignore[override]
        """Execute one tool call and return the observation + reward."""
        if self._done:
            return self._terminal_obs("Episode already finished.")

        self._state.step_count += 1
        step = self._state.step_count
        max_steps = self._task["max_steps"]

        # --- validate tool name ---
        if action.tool not in VALID_TOOLS:
            reward = -0.05
            self._cumulative_reward += reward
            obs = SQLDebugObservation(
                tool_result=f"Unknown tool: '{action.tool}'.",
                tool_valid=False,
                task_description=self._task["description"],
                broken_sql=self._task["broken_sql"],
                step=step,
                cumulative_reward=self._cumulative_reward,
                error_hint=f"Valid tools are: {sorted(VALID_TOOLS)}",
                available_tables=self._get_table_names(),
                done=False,
                reward=reward,
            )
            return obs

        # --- dispatch tool ---
        tool_result, reward, done, error_hint = self._dispatch(action.tool, action.params)
        self._cumulative_reward += reward

        # Episode ends when solved OR step budget exhausted
        if not done and step >= max_steps:
            done = True

        self._done = done

        return SQLDebugObservation(
            tool_result=tool_result,
            tool_valid=True,
            task_description=self._task["description"],
            broken_sql=self._task["broken_sql"],
            step=step,
            cumulative_reward=self._cumulative_reward,
            error_hint=error_hint,
            available_tables=self._get_table_names(),
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _dispatch(
        self, tool: str, params: Dict[str, Any]
    ) -> Tuple[str, float, bool, Optional[str]]:
        """Route to the correct tool handler. Returns (result, reward, done, error_hint)."""
        if tool == "list_tables":
            return self._tool_list_tables()
        elif tool == "inspect_schema":
            return self._tool_inspect_schema(params)
        elif tool == "run_query":
            return self._tool_run_query(params)
        elif tool == "validate_fix":
            return self._tool_validate_fix(params)
        # Should never reach here due to earlier check
        return "Unknown tool.", -0.05, False, None

    def _tool_list_tables(self) -> Tuple[str, float, bool, Optional[str]]:
        tables = self._get_table_names()
        result = f"Tables in database: {tables}"
        # Small reward for using a valid tool
        return result, 0.10, False, None

    def _tool_inspect_schema(
        self, params: Dict[str, Any]
    ) -> Tuple[str, float, bool, Optional[str]]:
        table_name = params.get("table_name", "").strip()
        if not table_name:
            return (
                "Missing parameter: table_name",
                -0.05,
                False,
                "Provide params={'table_name': '<name>'}",
            )

        try:
            cursor = self._conn.execute(f"PRAGMA table_info({table_name})")
            rows = cursor.fetchall()
        except Exception as e:
            return f"Error inspecting schema: {e}", -0.05, False, None

        if not rows:
            return (
                f"Table '{table_name}' not found.",
                -0.05,
                False,
                f"Available tables: {self._get_table_names()}",
            )

        columns = [
            {"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]}
            for r in rows
        ]
        result = f"Schema for '{table_name}':\n{json.dumps(columns, indent=2)}"

        # Bonus reward for inspecting schema before running queries — good diagnosis
        reward = 0.20 if not self._schema_inspected else 0.10
        self._schema_inspected = True
        return result, reward, False, None

    def _tool_run_query(
        self, params: Dict[str, Any]
    ) -> Tuple[str, float, bool, Optional[str]]:
        sql = params.get("sql", "").strip()
        if not sql:
            return (
                "Missing parameter: sql",
                -0.05,
                False,
                "Provide params={'sql': '<your sql>'}",
            )

        try:
            cursor = self._conn.execute(sql)
            rows = cursor.fetchall()
        except Exception as e:
            return (
                f"Query error: {e}",
                0.05,   # small reward — at least they tried running a query
                False,
                "Fix the SQL syntax or column/table names and try again.",
            )

        # Query ran successfully
        result = f"Query returned {len(rows)} row(s):\n{json.dumps(rows, indent=2)}"
        reward = 0.30
        self._valid_run_attempted = True
        return result, reward, False, None

    def _tool_validate_fix(
        self, params: Dict[str, Any]
    ) -> Tuple[str, float, bool, Optional[str]]:
        """
        The agent submits its fixed SQL. We run it and compare against expected output.
        This is the grader — it determines final score.
        """
        fixed_sql = params.get("fixed_sql", "").strip()
        if not fixed_sql:
            return (
                "Missing parameter: fixed_sql",
                -0.05,
                False,
                "Provide params={'fixed_sql': '<corrected sql>'}",
            )

        # Run the fixed query
        try:
            cursor = self._conn.execute(fixed_sql)
            actual_rows = cursor.fetchall()
        except Exception as e:
            return (
                f"Fixed query failed to execute: {e}",
                0.10,   # tried to fix, partial credit
                False,
                "The fixed query has errors. Try run_query to test it first.",
            )

        expected_rows = self._task["expected_rows"]

        # Compare results — order-insensitive for fairness
        actual_set = set(tuple(r) for r in actual_rows)
        expected_set = set(tuple(r) for r in expected_rows)

        if actual_set == expected_set:
            self._solved = True
            return (
                f"✓ Correct! Query returns exactly the expected {len(expected_rows)} row(s).\n"
                f"Rows: {json.dumps(actual_rows, indent=2)}",
                1.00,
                True,   # episode done
                None,
            )

        # Partial match — some rows correct
        correct_rows = actual_set & expected_set
        partial_ratio = len(correct_rows) / max(len(expected_set), 1)
        partial_reward = 0.50 * partial_ratio

        return (
            f"✗ Not quite. Got {len(actual_rows)} row(s), expected {len(expected_rows)}.\n"
            f"Your output: {json.dumps(actual_rows, indent=2)}\n"
            f"Partial match: {len(correct_rows)}/{len(expected_set)} rows correct.",
            partial_reward,
            False,   # let agent keep trying
            "Check your WHERE/HAVING conditions and JOIN columns.",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_table_names(self) -> List[str]:
        if not self._conn:
            return []
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r[0] for r in cursor.fetchall()]

    def _terminal_obs(self, msg: str) -> SQLDebugObservation:
        return SQLDebugObservation(
            tool_result=msg,
            tool_valid=False,
            task_description=self._task["description"],
            broken_sql=self._task["broken_sql"],
            step=self._state.step_count,
            cumulative_reward=self._cumulative_reward,
            available_tables=self._get_table_names(),
            done=True,
            reward=0.0,
        )