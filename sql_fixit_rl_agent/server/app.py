"""
FastAPI application for the SQL Debug Environment.

Exposes the SQLDebugEnvironment over HTTP endpoints compatible with EnvClient.
Supports task selection via query parameter: ?task=easy|medium|hard

Endpoints:
    POST /reset          Reset the environment (optionally pass {"task": "medium"})
    POST /step           Execute a tool call action
    GET  /state          Get current environment state
    GET  /schema         Get action/observation schemas
    GET  /tasks          List available tasks with descriptions
    WS   /ws             WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import SQLDebugAction, SQLDebugObservation
    from .sql_debug_environment import SQLDebugEnvironment
except ModuleNotFoundError:
    from sql_fixit_rl_agent.models import SQLDebugAction, SQLDebugObservation
    from sql_fixit_rl_agent.server.sql_debug_environment import SQLDebugEnvironment

# ---------------------------------------------------------------------------
# App setup — pass a factory function (callable) to create_app
# ---------------------------------------------------------------------------

# Default task from env var (useful for running different tasks in CI)
DEFAULT_TASK = os.getenv("SQL_DEBUG_TASK", "easy")


def _env_factory() -> SQLDebugEnvironment:
    """Factory that returns a fresh SQLDebugEnvironment instance."""
    return SQLDebugEnvironment(task_name=DEFAULT_TASK)


app = create_app(
    _env_factory,
    SQLDebugAction,
    SQLDebugObservation,
    env_name="sql-debug",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution via [project.scripts].
    
    Usage: main()  # defaults to 0.0.0.0:8000
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def _cli_entry() -> None:
    """CLI entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="SQL Debug Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--task",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Task difficulty to run",
    )
    args = parser.parse_args()
    os.environ["SQL_DEBUG_TASK"] = args.task
    main(host=args.host, port=args.port)


if __name__ == "__main__":
    _cli_entry()