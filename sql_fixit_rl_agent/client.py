"""SQL FixIt RL Agent Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SQLDebugAction, SQLDebugObservation


class SQLDebugEnv(EnvClient[SQLDebugAction, SQLDebugObservation, State]):
    """
    Client for the SQL Debug Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with SQLDebugEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.task_description)
        ...     print(result.observation.broken_sql)
        ...
        ...     # List tables first
        ...     result = env.step(SQLDebugAction(tool="list_tables", params={}))
        ...     print(result.observation.tool_result)
        ...
        ...     # Inspect schema
        ...     result = env.step(SQLDebugAction(
        ...         tool="inspect_schema",
        ...         params={"table_name": "customers"}
        ...     ))
        ...
        ...     # Submit fix
        ...     result = env.step(SQLDebugAction(
        ...         tool="validate_fix",
        ...         params={"fixed_sql": "SELECT name, email FROM customers WHERE order_year = 2024;"}
        ...     ))
        ...     print(result.reward)   # 1.0 if correct

    Example with Docker:
        >>> env = SQLDebugEnv.from_docker_image("sql-debug-env:latest")
        >>> try:
        ...     result = env.reset()
        ...     result = env.step(SQLDebugAction(tool="list_tables", params={}))
        ... finally:
        ...     env.close()
    """

    def _step_payload(self, action: SQLDebugAction) -> Dict:
        return {
            "tool": action.tool,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SQLDebugObservation]:
        obs_data = payload.get("observation", {})
        observation = SQLDebugObservation(
            tool_result=obs_data.get("tool_result", ""),
            tool_valid=obs_data.get("tool_valid", True),
            task_description=obs_data.get("task_description", ""),
            broken_sql=obs_data.get("broken_sql", ""),
            step=obs_data.get("step", 0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            error_hint=obs_data.get("error_hint"),
            available_tables=obs_data.get("available_tables", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )