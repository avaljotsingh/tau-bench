"""Phase 1 — harvest gpt-4o failures to build a difficulty-controlled hard set.

gpt-4o is saturated on easy retail tasks (45/50), so we restore headroom by finding the tasks it actually
fails, keeping its low-noise advantage. Runs a larger train-split pool at temp 0 and records failures (with
complete env_actions for replay-faithful failing trajectories).
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

START, END = 50, 250            # 200 more train tasks
OUT = "experiments/phase1"
CKPT = os.path.join(OUT, "gpt4o_harvest.json")

cfg = RunConfig(
    model_provider="openai", user_model_provider="openai", model="none", user_model="none",
    num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="train",
    start_index=0, end_index=-1, task_ids=list(range(START, END)), log_dir=OUT, max_concurrency=8,
    seed=10, shuffle=0, user_strategy="llm", few_shot_displays_path=None,
    mcp_server="mcp/retail_server.py", ckpt_path=CKPT, new_func=None)

res = tau_run(cfg)
passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
failures = sorted(r.task_id for r in res if r.reward < 1 - 1e-6)
# combine with the 0-49 base failures for the full hard set
prev = json.load(open(os.path.join(OUT, "gpt4o_base_report.json")))["base_failures"]
hard = sorted(set(prev) | set(failures))
report = {"model": "gpt-4o_2024-11-20", "pool": f"{START}-{END-1}", "passed": passed, "n": END - START,
          "harvest_failures": failures, "combined_hard_set": hard, "hard_set_size": len(hard)}
with open(os.path.join(OUT, "gpt4o_harvest_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\n=== harvest {START}-{END-1}: {passed}/{END-START} passed | new failures={failures} ===")
print(f"=== COMBINED HARD SET ({len(hard)} tasks): {hard} ===")
