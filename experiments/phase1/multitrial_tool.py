"""Multi-trial noise check: is list_user_orders_with_status really harmful, or single-trial noise?

base vs base+list_user_orders_with_status, 50 tasks x 3 trials, gpt-4o-mini temp 0. Reports mean pass-rate
and per-task deltas; a task counts as robustly changed only if the sign holds in >=2/3 trials.
"""
import os, sys, json, ast
from collections import defaultdict
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["TRAPI_MODEL_VERSION"] = "2024-07-18"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

K = 3
N = 50
OUT = "experiments/phase1/multitrial_tool"
os.makedirs(OUT, exist_ok=True)
BASE_SERVER = "mcp/retail_server.py"
TOOL = "list_user_orders_with_status"
TOOL_SERVER = os.path.join(OUT, "base_plus_tool.py")

# build base + just the one tool
aug = open("experiments/phase1/mined_from_gpt4o/augmented_mined.py", encoding="latin-1").read()
tool_src = next(ast.get_source_segment(aug, n) for n in ast.parse(aug).body
                if isinstance(n, ast.FunctionDef) and n.name == TOOL)
b = open(BASE_SERVER, encoding="latin-1").read().splitlines()
at = next((i for i, ln in enumerate(b) if ln.strip().startswith("if __name__")), len(b))
open(TOOL_SERVER, "w", encoding="utf-8").write(
    "\n".join(b[:at]).rstrip() + "\n\n@mcp.tool()\n" + tool_src.strip() + "\n\n" + "\n".join(b[at:]) + "\n")


def cfg(server, ckpt):
    return RunConfig(model_provider="openai", user_model_provider="openai", model="none", user_model="none",
        num_trials=K, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="train",
        start_index=0, end_index=-1, task_ids=list(range(N)), log_dir=OUT, max_concurrency=8, seed=10,
        shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=server, ckpt_path=ckpt, new_func=None)


def rates(res):
    by = defaultdict(list)
    for r in res:
        by[r.task_id].append(1.0 if r.reward >= 1 - 1e-6 else 0.0)
    return {t: sum(v) / len(v) for t, v in by.items()}


print(f"=== BASE x{K} ===", flush=True)
base = rates(tau_run(cfg(BASE_SERVER, os.path.join(OUT, "base.json"))))
print(f"=== BASE + {TOOL} x{K} ===", flush=True)
tool = rates(tau_run(cfg(TOOL_SERVER, os.path.join(OUT, "tool.json"))))

deltas = {t: round(tool.get(t, 0) - base.get(t, 0), 3) for t in base}
bm = round(sum(base.values()) / N, 3); tm = round(sum(tool.values()) / N, 3)
robust_down = sorted(t for t, d in deltas.items() if d <= -0.66)
robust_up = sorted(t for t, d in deltas.items() if d >= 0.66)
rep = {"tool": TOOL, "n_tasks": N, "n_trials": K, "base_mean": bm, "tool_mean": tm,
       "mean_delta": round(tm - bm, 3), "robustly_worsened": robust_down, "robustly_improved": robust_up,
       "per_task_delta": {str(t): deltas[t] for t in sorted(deltas) if deltas[t] != 0}}
json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=2)
print(f"\n=== base {bm} vs +{TOOL} {tm} | delta {tm-bm:+.3f} ===")
print(f"robustly worsened (>=2/3 trials): {robust_down}")
print(f"robustly improved  (>=2/3 trials): {robust_up}")
