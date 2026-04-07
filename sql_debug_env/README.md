---
title: SQL Debug Environment
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# SQL Debug Environment

An OpenEnv environment where an AI agent diagnoses and fixes broken SQL queries
by calling tools against a simulated SQLite database.

**Real-world motivation:** SQL debugging is a daily task for data engineers,
analysts, and backend developers. A well-trained agent could power IDE plugins,
CI/CD query validators, or automated code review tools.

---

## Tasks

| Task   | Bug Type                                     | Max Steps |
|--------|----------------------------------------------|-----------|
| easy   | Single syntax error (missing keyword)        | 6         |
| medium | Wrong column name in a JOIN condition        | 10        |
| hard   | Three simultaneous bugs (aggregate + GROUP BY + HAVING) | 15 |

---

## Action Space

The agent submits one tool call per step as a `SQLDebugAction`:

```python
SQLDebugAction(
    tool="inspect_schema",          # one of the 4 tools below
    params={"table_name": "orders"} # tool-specific parameters
)
```

### Available Tools

| Tool            | Params                          | Description                              |
|-----------------|---------------------------------|------------------------------------------|
| `list_tables`   | `{}`                            | List all tables in the database          |
| `inspect_schema`| `{"table_name": "<n>"}`         | Show columns + types for a table         |
| `run_query`     | `{"sql": "<sql>"}`              | Execute any SQL, returns rows or error   |
| `validate_fix`  | `{"fixed_sql": "<corrected>"}`  | Submit fixed SQL — grader call           |

---

## Observation Space

Each step returns a `SQLDebugObservation`:

| Field               | Type           | Description                                      |
|---------------------|----------------|--------------------------------------------------|
| `tool_result`       | str            | Output from the last tool call                   |
| `tool_valid`        | bool           | Whether the tool call was valid                  |
| `task_description`  | str            | Full task description (always visible)           |
| `broken_sql`        | str            | The original broken query (always visible)       |
| `step`              | int            | Current step number                              |
| `cumulative_reward` | float          | Total reward accumulated this episode            |
| `available_tables`  | List[str]      | Tables in the database                           |
| `error_hint`        | Optional[str]  | Hint when something goes wrong                   |
| `done`              | bool           | Whether the episode has ended                    |
| `reward`            | float          | Reward for the last step                         |

---

## Reward Function

Rewards are shaped to give signal at every step — not just on final success:

| Event                                          | Reward  |
|------------------------------------------------|---------|
| `list_tables` called                           | +0.10   |
| `inspect_schema` called (first time)           | +0.20   |
| `inspect_schema` called (subsequent)           | +0.10   |
| `run_query` returns rows (valid syntax)        | +0.30   |
| `run_query` raises error (tried but failed)    | +0.05   |
| `validate_fix` — query runs, partial row match | +0.00–0.50 |
| `validate_fix` — exact match ✓                 | +1.00   |
| Unknown tool / missing params                  | −0.05   |

---

## Quick Start

```python
from sql_debug_env import SQLDebugAction, SQLDebugEnv

env = SQLDebugEnv.from_docker_image("sql-debug-env:latest")
try:
    result = env.reset()
    print(result.observation.task_description)
    print(result.observation.broken_sql)

    # Step 1: list tables
    result = env.step(SQLDebugAction(tool="list_tables", params={}))
    print(result.observation.tool_result)

    # Step 2: inspect schema
    result = env.step(SQLDebugAction(
        tool="inspect_schema",
        params={"table_name": "customers"}
    ))

    # Step 3: submit fix
    result = env.step(SQLDebugAction(
        tool="validate_fix",
        params={"fixed_sql": "SELECT name, email FROM customers WHERE order_year = 2024;"}
    ))
    print(result.reward)  # 1.0 if correct
finally:
    env.close()
```

---

## Running the Baseline Inference Script

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LOCAL_IMAGE_NAME=sql-debug-env:latest

# Build Docker image first
docker build -t sql-debug-env:latest -f Dockerfile .

# Run inference across all 3 tasks
python inference.py
```

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task   | Score | Steps |
|--------|-------|-------|
| easy   | 0.85  | 4     |
| medium | 0.72  | 7     |
| hard   | 0.51  | 12    |

---

## Project Structure

```
sql-debug-env/
├── Dockerfile                  # Container definition
├── inference.py                # Baseline inference script (root level, mandatory)
└── sql_debug_env/
    ├── __init__.py             # Exports SQLDebugEnv, SQLDebugAction, SQLDebugObservation
    ├── client.py               # WebSocket client
    ├── models.py               # SQLDebugAction + SQLDebugObservation Pydantic models
    ├── openenv.yaml            # OpenEnv manifest
    ├── pyproject.toml          # Project metadata
    └── server/
        ├── __init__.py
        ├── app.py              # FastAPI application
        ├── sql_debug_environment.py # Core environment + tool logic + reward shaping
        └── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install openenv-core[core] fastapi uvicorn openai

# Run locally
uvicorn sql_debug_env.server.app:app --reload --host 0.0.0.0 --port 8000

# Run hard task
SQL_DEBUG_TASK=hard uvicorn sql_debug_env.server.app:app --host 0.0.0.0 --port 8000
```