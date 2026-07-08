"""Base-vs-augmented library step-efficiency A/B experiment.

Runs the SAME retail test tasks twice with the plain tool-calling agent:
  - Arm A: base library      (mcp/retail_server.py)
  - Arm B: augmented library (base + generated composite tools)
Measures reward (should stay ~1.0, it's saturated) and the real signal:
number of tool calls (env actions) to solve each task, plus how often the
agent actually uses the new composite tools.
"""
import os, json
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-5")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-11-20")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-5_2025-08-07")

from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

TASK_IDS = list(range(80, 100))            # retail test split, 20 tasks (base solves 20/20)
COMPOSITES = {"authenticate_user", "resolve_user_id", "list_user_orders", "calculate_item_price_difference"}
OUT_DIR = "experiments/ab_experiment"
os.makedirs(OUT_DIR, exist_ok=True)

def make_cfg(mcp_server, ckpt):
    return RunConfig(
        model_provider="openai", user_model_provider="openai",
        model="none", user_model="none",
        num_trials=1, env="retail", agent_strategy="tool-calling",
        temperature=0.2, task_split="test",
        start_index=0, end_index=-1, task_ids=TASK_IDS,
        log_dir=OUT_DIR, max_concurrency=6, seed=10, shuffle=0,
        user_strategy="llm", few_shot_displays_path=None,
        mcp_server=mcp_server, ckpt_path=ckpt, new_func=None,
    )

def tool_calls(traj):
    """Count env actions = number of 'tool' role messages."""
    return sum(1 for m in traj if isinstance(m, dict) and m.get("role") == "tool")

def composite_calls(traj):
    c = 0
    for m in traj:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("name") in COMPOSITES:
            c += 1
    return c

def summarize(results):
    n = len(results)
    rewards = [r.reward for r in results]
    steps = [tool_calls(r.traj) for r in results]
    comp = [composite_calls(r.traj) for r in results]
    return {
        "n": n,
        "avg_reward": round(sum(rewards) / n, 4),
        "solved": sum(1 for x in rewards if x >= 1 - 1e-6),
        "avg_tool_calls": round(sum(steps) / n, 2),
        "total_tool_calls": sum(steps),
        "tasks_using_composite": sum(1 for c in comp if c > 0),
        "total_composite_calls": sum(comp),
    }

print("=== ARM A: BASE library ===", flush=True)
base_res = tau_run(make_cfg("mcp/retail_server.py", os.path.join(OUT_DIR, "base.json")))
print("=== ARM B: AUGMENTED library ===", flush=True)
aug_res = tau_run(make_cfg("mcp/augmented_retail_server.py", os.path.join(OUT_DIR, "augmented.json")))

base_s, aug_s = summarize(base_res), summarize(aug_res)
report = {"base": base_s, "augmented": aug_s,
          "tool_call_reduction_pct": round(100 * (base_s["avg_tool_calls"] - aug_s["avg_tool_calls"]) / base_s["avg_tool_calls"], 1) if base_s["avg_tool_calls"] else 0}
with open(os.path.join(OUT_DIR, "ab_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n================ A/B RESULT ================")
print(json.dumps(report, indent=2))
