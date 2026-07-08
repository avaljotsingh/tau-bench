"""Multi-trial validation: is the filter-surviving tool's benefit REAL or noise?

Single-trial measurement has ~+-16% run-to-run variance, larger than any tool effect.
Multi-trial averages each task over N trials so the per-task pass-RATE estimates its
true success probability, and the base-vs-tool delta rises above the noise floor.

Arms (gpt-4o-mini agent, both): base library; base + find_order_by_item_with_tracking.
50 retail tasks x N trials each. Reports per-task pass-rate delta + robust changes.
Usage: python experiments/run_multitrial.py [n_trials]
"""
import os, sys, json, ast
from collections import defaultdict
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-07-18")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-4o-mini_2024-07-18")

from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N = 50
TASK_IDS = list(range(0, N))
OUT_DIR = "experiments/multitrial"
os.makedirs(OUT_DIR, exist_ok=True)
BASE_SERVER = "mcp/retail_server.py"
SURVIVOR = "find_order_by_item_with_tracking"
SURV_SERVER = os.path.join(OUT_DIR, "base_plus_survivor.py")
NUDGE = (" IMPORTANT: A higher-level composite tool may be available that bundles a multi-step "
         "lookup. When it fits, prefer a single call to it instead of the lower-level calls, and "
         "don't redo the same work afterward.")

# build base + survivor server
aug_src = open("experiments/weak_experiment/augmented.py", encoding="latin-1").read()
surv_src = next(ast.get_source_segment(aug_src, n) for n in ast.parse(aug_src).body
                if isinstance(n, ast.FunctionDef) and n.name == SURVIVOR)
b = open(BASE_SERVER, encoding="latin-1").read().splitlines()
at = next((i for i, ln in enumerate(b) if ln.strip().startswith("if __name__")), len(b))
open(SURV_SERVER, "w", encoding="utf-8").write(
    "\n".join(b[:at]).rstrip() + "\n\n@mcp.tool()\n" + surv_src.strip() + "\n\n" + "\n".join(b[at:]) + "\n")

def cfg(server, ckpt):
    return RunConfig(model_provider="openai", user_model_provider="openai", model="none", user_model="none",
        num_trials=N_TRIALS, env="retail", agent_strategy="tool-calling", temperature=0.2, task_split="test",
        start_index=0, end_index=-1, task_ids=TASK_IDS, log_dir=OUT_DIR, max_concurrency=8, seed=10,
        shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=server, ckpt_path=ckpt, new_func=None)

def pass_rates(results):
    by_task = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(1.0 if r.reward >= 1 - 1e-6 else 0.0)
    return {t: sum(v) / len(v) for t, v in by_task.items()}

print(f"=== ARM: BASE (gpt-4o-mini x{N_TRIALS} trials) ===", flush=True)
os.environ["AGENT_PROMPT_SUFFIX"] = ""
base_rates = pass_rates(tau_run(cfg(BASE_SERVER, os.path.join(OUT_DIR, "base.json"))))
print(f"=== ARM: BASE + {SURVIVOR} (x{N_TRIALS}) ===", flush=True)
os.environ["AGENT_PROMPT_SUFFIX"] = NUDGE
surv_rates = pass_rates(tau_run(cfg(SURV_SERVER, os.path.join(OUT_DIR, "survivor.json"))))
os.environ["AGENT_PROMPT_SUFFIX"] = ""

deltas = {t: round(surv_rates.get(t, 0) - base_rates.get(t, 0), 3) for t in base_rates}
base_acc = round(sum(base_rates.values()) / N, 3)
surv_acc = round(sum(surv_rates.values()) / N, 3)
robust_up = sorted([t for t, d in deltas.items() if d >= 0.6])     # improved in >=60% of trials
robust_down = sorted([t for t, d in deltas.items() if d <= -0.6])  # worsened in >=60% of trials
report = {
    "model": "gpt-4o-mini", "n_tasks": N, "n_trials": N_TRIALS,
    "base_mean_passrate": base_acc, "survivor_mean_passrate": surv_acc,
    "mean_delta": round(surv_acc - base_acc, 3),
    "robustly_improved_tasks": robust_up, "robustly_worsened_tasks": robust_down,
    "per_task_delta": {str(t): deltas[t] for t in sorted(deltas) if deltas[t] != 0},
}
json.dump(report, open(os.path.join(OUT_DIR, "report.json"), "w"), indent=2)
print("\n================ MULTI-TRIAL RESULT ================")
print(json.dumps(report, indent=2))
