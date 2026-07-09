"""After S1-S3 scoring, deploy ONLY the survivor tools on held-out test tasks.

Compares: base vs S3-filtered teacher_student vs S3-filtered self_improve.
"""
import os, sys, json
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["TRAPI_MODEL_VERSION"] = "2024-07-18"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

TASK_IDS = list(range(80, 100))
N = len(TASK_IDS)
OUT = "experiments/scoring/holdout_filtered"
os.makedirs(OUT, exist_ok=True)

OFFLINE_DIR = "experiments/scoring/offline"

CONFIGS = [
    {
        "label": "base",
        "server": "mcp/retail_server.py",
    },
    {
        "label": "teacher_student_filtered",
        "server": os.path.join(OFFLINE_DIR, "teacher_student_filtered_server.py"),
    },
    {
        "label": "self_improve_filtered",
        "server": os.path.join(OFFLINE_DIR, "self_improve_filtered_server.py"),
    },
]

# Also include unfiltered for comparison
CONFIGS.extend([
    {
        "label": "teacher_student_unfiltered",
        "server": "experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py",
    },
    {
        "label": "self_improve_unfiltered",
        "server": "experiments/phase1/self_improve/augmented_repaired.py",
    },
])

results = {}
for c in CONFIGS:
    label = c["label"]
    server = c["server"]

    if not os.path.exists(server):
        print(f"\nSKIPPING {label}: server not found at {server}")
        print("(Run scoring/s1_s2_s3_pipeline.py first to generate filtered servers)")
        continue

    ckpt = os.path.join(OUT, f"{label}_test.json")
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Server: {server}")
    print(f"{'='*60}")

    cfg = RunConfig(
        model_provider="openai", user_model_provider="openai", model="none", user_model="none",
        num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="test",
        start_index=0, end_index=-1, task_ids=TASK_IDS, log_dir=OUT, max_concurrency=8, seed=10,
        shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=server,
        ckpt_path=ckpt, new_func=None)

    res = tau_run(cfg)
    passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
    failures = sorted(r.task_id for r in res if r.reward < 1 - 1e-6)
    results[label] = {"passed": passed, "n": N, "pass_rate": round(passed / N, 3), "failures": failures}
    print(f"\n>>> {label}: {passed}/{N} (failures: {failures})")

# Report
base_score = results.get("base", {}).get("passed", 0)
report = {
    "experiment": "S3-filtered holdout test (tasks 80-99)",
    "model": "gpt-4o-mini_2024-07-18",
    "results": results,
    "comparison": {
        label: {
            "passed": r["passed"],
            "delta_vs_base": r["passed"] - base_score,
        }
        for label, r in results.items()
    }
}
with open(os.path.join(OUT, "filtered_holdout_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*60}")
print(f"FILTERED HOLDOUT RESULTS (tasks 80-99)")
print(f"{'='*60}")
for label, r in results.items():
    delta = r["passed"] - base_score
    sign = "+" if delta > 0 else ""
    print(f"  {label:40s} {r['passed']}/{N} ({sign}{delta})")
print(f"{'='*60}")
