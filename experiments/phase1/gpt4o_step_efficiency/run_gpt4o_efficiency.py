"""Deploy teacher-student tools to the STRONGER gpt-4o agent (saturated, 45/50 base).

gpt-4o doesn't need accuracy help — it already gets 45/50. The hypothesis:
composite tools should still reduce STEP COUNT on passing tasks, even when
the model is smart enough to solve everything with base tools.

Runs base + augmented on test split (80-99) to measure step efficiency.
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

TASK_IDS = list(range(80, 100))
N = len(TASK_IDS)
OUT = "experiments/phase1/gpt4o_step_efficiency"
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    {
        "label": "gpt4o_base",
        "server": "mcp/retail_server.py",
        "ckpt": os.path.join(OUT, "gpt4o_base_test.json"),
    },
    {
        "label": "gpt4o_teacher_student",
        "server": "experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py",
        "ckpt": os.path.join(OUT, "gpt4o_teacher_student_test.json"),
    },
]

results = {}
for c in CONFIGS:
    label = c["label"]
    ckpt = c["ckpt"]
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    cfg = RunConfig(
        model_provider="openai", user_model_provider="openai", model="none", user_model="none",
        num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="test",
        start_index=0, end_index=-1, task_ids=TASK_IDS, log_dir=OUT, max_concurrency=8, seed=10,
        shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=c["server"],
        ckpt_path=ckpt, new_func=None)

    res = tau_run(cfg)
    passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
    failures = sorted(r.task_id for r in res if r.reward < 1 - 1e-6)
    results[label] = {"passed": passed, "n": N, "failures": failures}
    print(f"\n>>> {label}: {passed}/{N} (failures: {failures})")

# Step efficiency from checkpoint files
print(f"\n{'='*60}")
print("STEP EFFICIENCY ANALYSIS")
print(f"{'='*60}")
for c in CONFIGS:
    label = c["label"]
    recs = json.load(open(c["ckpt"]))
    passing = [r for r in recs if r["reward"] >= 1 - 1e-6]
    calls = {}
    for r in passing:
        ea = (r.get("records") or {}).get("env_actions", [])
        if ea:
            calls[r["task_id"]] = len([a for a in ea if a["name"] != "respond"])
    avg = sum(calls.values()) / len(calls) if calls else 0
    results[label]["avg_calls"] = round(avg, 2)
    results[label]["total_calls"] = sum(calls.values())
    results[label]["per_task_calls"] = calls
    print(f"  {label:30s}  pass={results[label]['passed']}/{N}  avg_calls={avg:.1f}  total={sum(calls.values())}")

# Per-task comparison
b_calls = results["gpt4o_base"].get("per_task_calls", {})
t_calls = results["gpt4o_teacher_student"].get("per_task_calls", {})
common = sorted(set(b_calls.keys()) & set(t_calls.keys()))
if common:
    print(f"\n{'Task':>6s}  {'Base':>5s}  {'Aug':>5s}  {'Delta':>6s}")
    total_saved = 0
    for tid in common:
        d = b_calls[tid] - t_calls[tid]
        total_saved += d
        print(f"{tid:>6d}  {b_calls[tid]:>5d}  {t_calls[tid]:>5d}  {d:>+6d}")
    print(f"{'TOTAL':>6s}  {'':>5s}  {'':>5s}  {total_saved:>+6d}")

report = {
    "experiment": "gpt-4o step efficiency on held-out test",
    "model": "gpt-4o_2024-11-20",
    "task_split": "test",
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_task_calls"} for k, v in results.items()},
}
with open(os.path.join(OUT, "gpt4o_step_efficiency_report.json"), "w") as f:
    json.dump(report, f, indent=2)
