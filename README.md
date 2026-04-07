---
title: SQL FixIt RL Agent
emoji: 🐛
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
base_path: /docs
pinned: false
tags:
  - openenv
---

# SQL FixIt RL Agent

An OpenEnv environment where an AI agent diagnoses and fixes broken SQL queries by calling tools against a simulated SQLite database.

## Motivation

SQL debugging is a daily task for data engineers, analysts, and backend developers. A well-trained agent could power IDE plugins, CI/CD query validators, or automated code review tools. This environment simulates that real-world workflow — the agent receives a broken SQL query, explores the database schema via tools, identifies the bug, and submits a corrected query.

---

## Tasks

Three tasks with increasing difficulty, each containing a pre-populated SQLite database and a broken SQL query:

| Task   | Bug Description                                         | Difficulty |
|--------|---------------------------------------------------------|------------|
| **easy**   | Missing `FROM` keyword in a simple `SELECT` query          | Easy       |
| **medium** | Wrong column reference in a `JOIN` condition across 2 tables | Medium     |
| **hard**   | Three simultaneous bugs: wrong aggregate, missing `GROUP BY`, `WHERE` instead of `HAVING` | Hard       |

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

| Tool             | Parameters                      | Description                              |
|------------------|---------------------------------|------------------------------------------|
| `list_tables`    | `{}`                            | List all tables in the database          |
| `inspect_schema` | `{"table_name": "<name>"}`      | Show columns + types for a table         |
| `run_query`      | `{"sql": "<sql>"}`              | Execute any SQL, returns rows or error   |
| `validate_fix`   | `{"fixed_sql": "<corrected>"}`  | Submit fixed SQL — final grading call    |

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

| Event                                               | Reward  |
|-----------------------------------------------------|---------|
| `list_tables` called                                | +0.10   |
| `inspect_schema` called (first time)                | +0.20   |
| `inspect_schema` called (subsequent)                | +0.10   |
| `run_query` returns rows (valid syntax)             | +0.30   |
| `run_query` raises error (tried but failed)         | +0.05   |
| `validate_fix` — query runs, partial row match      | +0.00–0.50 |
| `validate_fix` — exact match ✓                     | +1.00   |
| Unknown tool / missing params                       | −0.05   |

---

## Quick Start

```bash
# Install dependencies
pip install openenv-core[core]

# Run the server locally
uvicorn sql_fixit_rl_agent.server.app:app --reload --host 0.0.0.0 --port 8000

# Run a specific task
SQL_DEBUG_TASK=hard uvicorn sql_fixit_rl_agent.server.app:app --host 0.0.0.0 --port 8000
```

### Using the Python Client

```python
from sql_fixit_rl_agent import SQLDebugAction, SQLDebugEnv

# Connect via Docker
env = SQLDebugEnv.from_docker_image("sql-debug-env:latest")
# Or connect to a running server
# env = SQLDebugEnv(base_url="http://localhost:8000")

try:
    result = await env.reset(task="easy")
    print(result.observation.task_description)

    # Step 1: list tables
    result = await env.step(SQLDebugAction(tool="list_tables", params={}))
    print(result.observation.tool_result)

    # Step 2: inspect schema
    result = await env.step(SQLDebugAction(
        tool="inspect_schema",
        params={"table_name": "customers"}
    ))

    # Step 3: submit fix
    result = await env.step(SQLDebugAction(
        tool="validate_fix",
        params={"fixed_sql": "SELECT name, email FROM customers WHERE order_year = 2024;"}
    ))
    print(result.reward)  # 1.0 if correct
finally:
    await env.close()
```

---

## Running the Baseline Inference Script

```bash
# Set environment variables
export HF_TOKEN=your_hf_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LOCAL_IMAGE_NAME=sql-debug-env:latest

# Build Docker image
docker build -t sql-debug-env:latest -f Dockerfile .

# Run inference across all 3 tasks
python inference.py
```

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task   | Score | Steps | Success |
|--------|-------|-------|---------|
| easy   | 0.520 | 3     | ✓       |
| medium | 0.486 | 5     | ✓       |
| hard   | 0.289 | 3     | ✓       |

**Overall average: 0.432**

All three tasks are solved correctly. The model is highly efficient, often solving in the minimum number of steps (list_tables → inspect_schema → validate_fix). Scores are normalized to [0, 1] using per-task maximum reward ceilings (easy: 2.5, medium: 3.5, hard: 4.5).

---

## Project Structure

```
sql-fixit-rl-agent/
├── Dockerfile                              # Container definition
├── inference.py                            # Baseline inference script (root level)
├── README.md                               # This file
├── .env.example                            # Environment variable template
├── validate-submission.sh                  # Pre-submission validation script
└── sql_fixit_rl_agent/
    ├── __init__.py                         # Exports SQLDebugEnv, SQLDebugAction, SQLDebugObservation
    ├── client.py                           # WebSocket client
    ├── models.py                           # Pydantic Action/Observation models
    ├── openenv.yaml                        # OpenEnv manifest
    ├── pyproject.toml                      # Project metadata
    ├── uv.lock                             # Dependency lock file
    └── server/
        ├── app.py                          # FastAPI application
        └── sql_debug_environment.py        # Core environment + tool logic + reward shaping
```

---

## Build & Deploy

```bash
# Build Docker image
docker build -t sql-debug-env:latest -f Dockerfile .

# Run locally
docker run -d -p 8000:8000 sql-debug-env:latest

# Test the server
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" -d '{}'

# Validate with openenv
cd sql_debug_env && openenv validate
```

---

## Hugging Face Space

This environment deploys to Hugging Face Spaces with the `openenv` tag.
The Space is accessible at: https://huggingface.co/spaces/Kamil-Shaikh-786/sql-debug-env

API documentation is available at `/docs` when the Space is running.
