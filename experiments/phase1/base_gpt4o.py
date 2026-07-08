"""Phase 1 — gpt-4o base run (headroom gate G1).

Measures gpt-4o's base pass rate on 50 retail tasks at temp 0 (low noise). Keep gpt-4o as the primary
capability-band agent iff base is in ~30-42/50 (real headroom, not saturated). Saves complete env_actions
so every trajectory is replay-faithful for the offline scorer.
"""
import os, sys, json
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_MODEL_NAME"] = "gpt-4o"
os.environ["TRAPI_MODEL_VERSION"] = "2024-11-20"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o_2024-11-20"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

N = 50
OUT = "experiments/phase1"
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "gpt4o_base.json")

cfg = RunConfig(
    model_provider="openai", user_model_provider="openai", model="none", user_model="none",
    num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="train",
    start_index=0, end_index=-1, task_ids=list(range(N)), log_dir=OUT, max_concurrency=8, seed=10,
    shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server="mcp/retail_server.py",
    ckpt_path=CKPT, new_func=None)

res = tau_run(cfg)
passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
failures = sorted(r.task_id for r in res if r.reward < 1 - 1e-6)
gate = "KEEP gpt-4o (has headroom)" if 30 <= passed <= 42 else \
       ("SATURATED -> consider harder tasks" if passed > 42 else "TOO WEAK -> unexpected")
report = {"model": "gpt-4o_2024-11-20", "temperature": 0.0, "n_tasks": N,
          "passed": passed, "pass_rate": round(passed / N, 3),
          "base_failures": failures, "G1_gate": gate}
with open(os.path.join(OUT, "gpt4o_base_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\n=== gpt-4o base: {passed}/{N} (pass_rate={passed/N:.3f}) | G1: {gate} ===")
print("base_failures:", failures)
