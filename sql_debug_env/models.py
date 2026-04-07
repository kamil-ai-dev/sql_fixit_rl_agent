"""
Data models for the SQL Debug Environment.

The agent interacts by calling tools (list_tables, inspect_schema, run_query, validate_fix).
Each step the agent picks one tool and provides parameters.
The observation returns the tool result + current task context.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SQLDebugAction(Action):
    """
    One tool call the agent wants to make.

    Available tools:
        list_tables       — no parameters needed
        inspect_schema    — params: {"table_name": "<name>"}
        run_query         — params: {"sql": "<sql string>"}
        validate_fix      — params: {"fixed_sql": "<corrected sql>"}
    """

    tool: str = Field(
        ...,
        description=(
            "Tool to call. One of: list_tables, inspect_schema, run_query, validate_fix"
        ),
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool call.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SQLDebugObservation(Observation):
    """
    What the agent sees after each tool call.
    """

    # Result of the tool that was just called
    tool_result: str = Field(
        default="",
        description="Output from the tool call (rows, schema, error message, etc.)",
    )

    # Was the last tool call valid (recognized tool + valid params)?
    tool_valid: bool = Field(
        default=True,
        description="False if the agent called an unknown tool or passed bad params.",
    )

    # Current task description — always visible so agent never loses context
    task_description: str = Field(
        default="",
        description="The task the agent must solve.",
    )

    # The broken SQL query the agent needs to fix
    broken_sql: str = Field(
        default="",
        description="The original broken SQL query.",
    )

    # Step number within this episode
    step: int = Field(default=0, description="Current step number.")

    # Cumulative reward so far this episode
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated in this episode so far.",
    )

    # Human-readable hint about what went wrong (only on errors)
    error_hint: Optional[str] = Field(
        default=None,
        description="Hint shown when tool call fails or query errors out.",
    )

    # List of tables available in this task's database
    available_tables: List[str] = Field(
        default_factory=list,
        description="Tables available in the simulated database.",
    )