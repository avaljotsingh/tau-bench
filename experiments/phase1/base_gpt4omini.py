"""Phase 1 (pivot) — gpt-4o-mini base run to harvest failures with replay-faithful trajectories.

gpt-4o is saturated on retail (can't produce failures cheaply). Since the deterministic offline scorer makes
agent noise irrelevant for tool SELECTION, we use gpt-4o-mini (lots of failures) as the agent and gpt-5 as the
generator. Temp 0 for reproducible failures. Saves complete env_actions so failing trajectories replay exactly.
"""
import os, sys, json
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["TRAPI_MODEL_VERSION"] = "2024-07-18"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

N = 50
OUT = "experiments/phase1"
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "gpt4omini_base.json")

cfg = RunConfig(
    model_provider="openai", user_model_provider="openai", model="none", user_model="none",
    num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="train",
    start_index=0, end_index=-1, task_ids=list(range(N)), log_dir=OUT, max_concurrency=8, seed=10,
    shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server="mcp/retail_server.py",
    ckpt_path=CKPT, new_func=None)

res = tau_run(cfg)
passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
failures = sorted(r.task_id for r in res if r.reward < 1 - 1e-6)
report = {"model": "gpt-4o-mini_2024-07-18", "temperature": 0.0, "n_tasks": N,
          "passed": passed, "pass_rate": round(passed / N, 3),
          "base_failures": failures, "n_failures": len(failures)}
with open(os.path.join(OUT, "gpt4omini_base_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\n=== gpt-4o-mini base: {passed}/{N} | {len(failures)} failures to target: {failures} ===")
