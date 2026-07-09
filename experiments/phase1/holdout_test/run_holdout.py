"""Held-out validation: run all three configurations on the TEST split (tasks 80-99).

Compares base vs teacher->student vs self-improve on 20 unseen tasks.
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

TASK_IDS = list(range(80, 100))
N = len(TASK_IDS)
OUT = "experiments/phase1/holdout_test"
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    {
        "label": "base",
        "server": "mcp/retail_server.py",
        "ckpt": os.path.join(OUT, "base_test.json"),
    },
    {
        "label": "teacher_student",
        "server": "experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py",
        "ckpt": os.path.join(OUT, "teacher_student_test.json"),
    },
    {
        "label": "self_improve",
        "server": "experiments/phase1/self_improve/augmented_repaired.py",
        "ckpt": os.path.join(OUT, "self_improve_test.json"),
    },
]

results = {}
for c in CONFIGS:
    label = c["label"]
    ckpt = c["ckpt"]
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"\n{'='*60}")
    print(f"Running: {label} (server: {c['server']})")
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
    results[label] = {"passed": passed, "n": N, "pass_rate": round(passed / N, 3), "failures": failures}
    print(f"\n>>> {label}: {passed}/{N} (failures: {failures})")

# Final comparison report
report = {
    "experiment": "held-out test split validation (tasks 80-99)",
    "model": "gpt-4o-mini_2024-07-18",
    "temperature": 0.0,
    "task_split": "test",
    "task_ids": TASK_IDS,
    "results": results,
    "summary": {
        "base": results["base"]["passed"],
        "teacher_student": results["teacher_student"]["passed"],
        "self_improve": results["self_improve"]["passed"],
        "teacher_student_delta": results["teacher_student"]["passed"] - results["base"]["passed"],
        "self_improve_delta": results["self_improve"]["passed"] - results["base"]["passed"],
    }
}
with open(os.path.join(OUT, "holdout_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*60}")
print(f"HELD-OUT TEST RESULTS (tasks 80-99, {N} tasks)")
print(f"{'='*60}")
print(f"  Base:              {results['base']['passed']}/{N}")
print(f"  Teacher->student:  {results['teacher_student']['passed']}/{N} ({'+' if results['teacher_student']['passed'] >= results['base']['passed'] else ''}{results['teacher_student']['passed'] - results['base']['passed']})")
print(f"  Self-improve:      {results['self_improve']['passed']}/{N} ({'+' if results['self_improve']['passed'] >= results['base']['passed'] else ''}{results['self_improve']['passed'] - results['base']['passed']})")
print(f"{'='*60}")
