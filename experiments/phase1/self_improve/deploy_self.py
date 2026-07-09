"""Self-improvement: deploy the verified self-mined tools back to gpt-4o-mini.

This is the headline comparison:
  - Teacher->student (gpt-4o -> mini): base 39 -> 46 (+7)
  - Self-improvement (mini -> mini):   base 39 -> ???
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
OUT = "experiments/phase1/self_improve"
CKPT = os.path.join(OUT, "gpt4omini_self_deploy.json")
SERVER = os.path.join(OUT, "augmented_repaired.py")

if not os.path.exists(SERVER):
    print(f"ERROR: repaired server not found at {SERVER}")
    print("Run repair_self.py first.")
    sys.exit(1)

# Clear checkpoint for a clean run
if os.path.exists(CKPT):
    os.remove(CKPT)

cfg = RunConfig(
    model_provider="openai", user_model_provider="openai", model="none", user_model="none",
    num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="train",
    start_index=0, end_index=-1, task_ids=list(range(N)), log_dir=OUT, max_concurrency=8, seed=10,
    shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=SERVER,
    ckpt_path=CKPT, new_func=None)

res = tau_run(cfg)
passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
failures = sorted(r.task_id for r in res if r.reward < 1 - 1e-6)

report = {
    "experiment": "self-improvement (gpt-4o-mini mines from own trajectories)",
    "model": "gpt-4o-mini_2024-07-18",
    "temperature": 0.0,
    "n_tasks": N,
    "passed": passed,
    "pass_rate": round(passed / N, 3),
    "failures": failures,
    "comparison": {
        "base": 39,
        "teacher_student (gpt4o->mini)": 46,
        "self_improve (mini->mini)": passed,
        "delta_vs_base": passed - 39,
        "delta_vs_teacher_student": passed - 46,
    }
}
with open(os.path.join(OUT, "self_improve_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*60}")
print(f"SELF-IMPROVEMENT RESULT: gpt-4o-mini + self-mined tools: {passed}/{N}")
print(f"  Base:              39/50")
print(f"  Teacher->student:  46/50 (+7)")
print(f"  Self-improve:      {passed}/50 ({'+' if passed >= 39 else ''}{passed - 39})")
print(f"{'='*60}")
print(f"Failures: {failures}")
