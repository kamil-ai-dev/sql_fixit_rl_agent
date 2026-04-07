"""
inference.py — SQL Debug Environment Baseline Inference Script
==============================================================

Runs a baseline LLM agent against all 3 tasks (easy, medium, hard).
Uses the OpenAI client to call the model.

Environment variables required:
    API_BASE_URL      LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          Hugging Face / API key
    LOCAL_IMAGE_NAME  Docker image name for the environment

STDOUT FORMAT (mandatory — do not change):
    [START] task=<task_name> env=sql-debug model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from sql_fixit_rl_agent import SQLDebugAction, SQLDebugEnv

# Load .env file if present
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME", "sql-debug-env:latest")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "sql-debug"

USE_DOCKER   = os.getenv("USE_DOCKER", "true").lower() == "true"

TASKS = ["easy", "medium", "hard"]

# Per-task step budgets (must stay under 20 min total runtime)
MAX_STEPS = {
    "easy":   15,
    "medium": 15,
    "hard":   15,
}

TEMPERATURE = 0.2   # Low temperature for deterministic tool calls
MAX_TOKENS  = 400

SUCCESS_THRESHOLD = 0.5   # cumulative_reward / max_possible >= this → success

# ---------------------------------------------------------------------------
# Stdout logging (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL debugger. You are given a broken SQL query and access
    to a simulated SQLite database via tools.

    Your goal: find and fix the bug(s) in the broken SQL query.

    Available tools (respond with EXACTLY this JSON format, nothing else):
      {"tool": "list_tables", "params": {}}
      {"tool": "inspect_schema", "params": {"table_name": "<name>"}}
      {"tool": "run_query", "params": {"sql": "<sql>"}}
      {"tool": "validate_fix", "params": {"fixed_sql": "<corrected sql>"}}

    Strategy:
      1. Call list_tables to see what tables exist.
      2. Call inspect_schema on relevant tables to check column names and types.
      3. Identify the bug(s) by comparing the broken query to the schema.
      4. Optionally test your fix with run_query before submitting.
      5. Call validate_fix with your corrected SQL to submit your answer.

    Rules:
      - Respond with ONLY a JSON object. No explanation, no markdown, no extra text.
      - Each response must be one tool call.
      - Do not guess column names — always inspect the schema first.
""").strip()


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

def build_user_prompt(
    obs_tool_result: str,
    task_description: str,
    broken_sql: str,
    step: int,
    history: List[str],
    cumulative_reward: float,
    error_hint: Optional[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "None yet."
    hint_block    = f"\nHint: {error_hint}" if error_hint else ""
    return textwrap.dedent(f"""
        TASK: {task_description}

        BROKEN SQL:
        {broken_sql}

        LAST TOOL RESULT (step {step}):
        {obs_tool_result}{hint_block}

        HISTORY:
        {history_block}

        Cumulative reward so far: {cumulative_reward:.2f}

        What is your next tool call? Respond with JSON only.
    """).strip()


def get_agent_action(
    client: OpenAI,
    obs,
    step: int,
    history: List[str],
) -> SQLDebugAction:
    """Call the LLM and parse its response into a SQLDebugAction."""
    user_prompt = build_user_prompt(
        obs_tool_result   = obs.tool_result,
        task_description  = obs.task_description,
        broken_sql        = obs.broken_sql,
        step              = step,
        history           = history,
        cumulative_reward = obs.cumulative_reward,
        error_hint        = obs.error_hint,
    )

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        return SQLDebugAction(
            tool   = data.get("tool", "list_tables"),
            params = data.get("params", {}),
        )

    except Exception as exc:
        print(f"[DEBUG] Model parse error: {exc}", flush=True)
        # Fallback: safe default action
        return SQLDebugAction(tool="list_tables", params={})


def action_str(action: SQLDebugAction) -> str:
    """Compact string representation for [STEP] log line."""
    if action.params:
        params_s = json.dumps(action.params, separators=(",", ":"))
        return f"{action.tool}({params_s})"
    return f"{action.tool}()"


# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> Dict[str, Any]:
    """Run one full episode for a given task. Returns summary dict."""
    max_steps = MAX_STEPS[task_name]
    rewards:   List[float] = []
    history:   List[str]   = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    if USE_DOCKER:
        env = await SQLDebugEnv.from_docker_image(IMAGE_NAME)
    else:
        env = SQLDebugEnv(base_url=ENV_URL)

    try:
        result = await env.reset(task=task_name)
        obs    = result.observation

        for step in range(1, max_steps + 1):
            # if result.done:
                # break

            action = get_agent_action(client, obs, step, history)
            result = await env.step(action)
            obs    = result.observation

            reward     = result.reward or 0.0
            done       = result.done
            error_hint = obs.error_hint

            rewards.append(reward)
            steps_taken = step

            log_step(
                step   = step,
                action = action_str(action),
                reward = reward,
                done   = done,
                error  = error_hint,
            )

            history.append(
                f"Step {step}: {action_str(action)} → reward={reward:+.2f} | {obs.tool_result[:80]}"
            )

            if done:
                break

        # Score = cumulative reward normalised to [0, 1]
        # Per-task max possible rewards (list_tables + inspect_schema×2 + run_query×N + validate_fix):
        #   easy:   0.10 + 0.30 + 0.90 + 1.00 = 2.30  → use 2.5
        #   medium: 0.10 + 0.30 + 1.80 + 1.00 = 3.20  → use 3.5
        #   hard:   0.10 + 0.30 + 2.70 + 1.00 = 4.10  → use 4.5
        MAX_POSSIBLE = {"easy": 2.5, "medium": 3.5, "hard": 4.5}[task_name]
        raw_score = obs.cumulative_reward / MAX_POSSIBLE
        score     = min(max(raw_score, 0.0), 1.0)
        success   = obs.cumulative_reward >= (MAX_POSSIBLE * SUCCESS_THRESHOLD)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main — run all 3 tasks sequentially
# ---------------------------------------------------------------------------

async def main() -> None:
    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = []

    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        summary = await run_task(client, task_name)
        results.append(summary)

    # Final summary to stdout
    print("\n[SUMMARY]", flush=True)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(
            f"  {status} {r['task']:8s}  score={r['score']:.3f}  steps={r['steps']}",
            flush=True,
        )

    overall = sum(r["score"] for r in results) / len(results)
    print(f"\n  Overall average score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())