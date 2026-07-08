"""Guided base-vs-augmented A/B: does usage guidance make the agent SUBSTITUTE
composite tools for multi-step sequences (fewer steps), instead of just adding them?

  - Arm A: base library,      plain agent (no nudge)
  - Arm B: augmented library, plain agent + a nudge to PREFER composites as replacements
Same 20 retail test tasks. Measures reward + tool-calls/task + composite adoption.
"""
import os, json
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-5")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-11-20")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-5_2025-08-07")

from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

TASK_IDS = list(range(80, 100))
COMPOSITES = {"authenticate_user", "resolve_user_id", "list_user_orders", "calculate_item_price_difference"}
OUT_DIR = "experiments/ab_guided"
os.makedirs(OUT_DIR, exist_ok=True)

NUDGE = (
    "\n\nIMPORTANT - efficiency: Higher-level composite tools are available: "
    "list_user_orders, authenticate_user, resolve_user_id, calculate_item_price_difference. "
    "Prefer a SINGLE call to one of these to REPLACE a sequence of lower-level calls whenever it fits "
    "(e.g. call list_user_orders once instead of get_user_details followed by repeated get_order_details; "
    "identify the user with authenticate_user OR resolve_user_id once, not both). "
    "Never call a composite and then redo the same work with lower-level tools. "
    "Use lower-level tools only when no composite applies."
)

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
    return sum(1 for m in traj if isinstance(m, dict) and m.get("role") == "tool")

def composite_calls(traj):
    return sum(1 for m in traj if isinstance(m, dict) and m.get("role") == "tool" and m.get("name") in COMPOSITES)

def summarize(results):
    n = len(results); rewards = [r.reward for r in results]
    steps = [tool_calls(r.traj) for r in results]; comp = [composite_calls(r.traj) for r in results]
    return {"n": n, "avg_reward": round(sum(rewards)/n, 4), "solved": sum(1 for x in rewards if x >= 1-1e-6),
            "avg_tool_calls": round(sum(steps)/n, 2), "total_tool_calls": sum(steps),
            "tasks_using_composite": sum(1 for c in comp if c > 0), "total_composite_calls": sum(comp)}

# Arm A: base, NO nudge
os.environ["AGENT_PROMPT_SUFFIX"] = ""
print("=== ARM A: BASE library (no nudge) ===", flush=True)
base_res = tau_run(make_cfg("mcp/retail_server.py", os.path.join(OUT_DIR, "base.json")))

# Arm B: augmented, WITH substitution nudge
os.environ["AGENT_PROMPT_SUFFIX"] = NUDGE
print("=== ARM B: AUGMENTED library (prefer-composites nudge) ===", flush=True)
aug_res = tau_run(make_cfg("mcp/augmented_retail_server.py", os.path.join(OUT_DIR, "augmented_guided.json")))
os.environ["AGENT_PROMPT_SUFFIX"] = ""

base_s, aug_s = summarize(base_res), summarize(aug_res)
report = {"base": base_s, "augmented_guided": aug_s,
          "tool_call_reduction_pct": round(100*(base_s["avg_tool_calls"]-aug_s["avg_tool_calls"])/base_s["avg_tool_calls"], 1) if base_s["avg_tool_calls"] else 0,
          "reward_delta": round(aug_s["avg_reward"]-base_s["avg_reward"], 3)}
with open(os.path.join(OUT_DIR, "ab_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("\n================ GUIDED A/B RESULT ================")
print(json.dumps(report, indent=2))
