"""S7: Step efficiency analysis — compare tool call counts across configurations.

No new runs needed — analyzes existing trajectory data.
"""
import json, os

OUT = "experiments/scoring/offline"
os.makedirs(OUT, exist_ok=True)

def analyze(label, path):
    recs = json.load(open(path))
    passing = [r for r in recs if r["reward"] >= 1 - 1e-6]
    per_task = {}
    for r in passing:
        ea = (r.get("records") or {}).get("env_actions", [])
        if ea:
            n = len([a for a in ea if a["name"] != "respond"])
            per_task[r["task_id"]] = n
    avg = sum(per_task.values()) / len(per_task) if per_task else 0
    return {
        "label": label,
        "n_passing": len(passing),
        "n_with_actions": len(per_task),
        "avg_calls": round(avg, 2),
        "total_calls": sum(per_task.values()),
        "per_task": per_task,
    }

configs = [
    ("base (train)", "experiments/phase1/gpt4omini_base.json"),
    ("teacher_student (train)", "experiments/phase1/gpt4omini_repaired_v3.json"),
    ("self_improve (train)", "experiments/phase1/self_improve/gpt4omini_self_deploy.json"),
    ("base (test)", "experiments/phase1/holdout_test/base_test.json"),
    ("teacher_student (test)", "experiments/phase1/holdout_test/teacher_student_test.json"),
    ("self_improve (test)", "experiments/phase1/holdout_test/self_improve_test.json"),
]

results = []
for label, path in configs:
    try:
        r = analyze(label, path)
        results.append(r)
        print(f"{r['label']:35s}  pass={r['n_passing']:2d}  avg_calls={r['avg_calls']:5.1f}  total={r['total_calls']}")
    except Exception as e:
        print(f"{label}: ERROR {e}")

# Per-task comparison for commonly passing tasks (train split)
print("\n--- Per-task call comparison (train, commonly passing tasks) ---")
base = {r["label"]: r for r in results}
if "base (train)" in base and "teacher_student (train)" in base:
    b = base["base (train)"]["per_task"]
    ts = base["teacher_student (train)"]["per_task"]
    si = base.get("self_improve (train)", {}).get("per_task", {})
    common = sorted(set(b.keys()) & set(ts.keys()))
    saved_ts, saved_si = 0, 0
    print(f"{'Task':>6s}  {'Base':>5s}  {'T→S':>5s}  {'Self':>5s}  {'Δ T→S':>6s}  {'Δ Self':>6s}")
    for tid in common:
        d_ts = b[tid] - ts.get(tid, b[tid])
        d_si = b[tid] - si.get(tid, b[tid])
        saved_ts += d_ts
        saved_si += d_si
        s_val = si.get(tid, None)
        s_str = str(s_val) if s_val is not None else "-"
        d_si_str = f"{d_si:>+6d}" if s_val is not None else "     -"
        print(f"{tid:>6d}  {b[tid]:>5d}  {ts[tid]:>5d}  {s_str:>5s}  {d_ts:>+6d}  {d_si_str}")
    print(f"{'TOTAL':>6s}  {'':>5s}  {'':>5s}  {'':>5s}  {saved_ts:>+6d}  {saved_si:>+6d}")

# Same for test split
print("\n--- Per-task call comparison (test, commonly passing tasks) ---")
if "base (test)" in base and "teacher_student (test)" in base:
    b = base["base (test)"]["per_task"]
    ts = base["teacher_student (test)"]["per_task"]
    si = base.get("self_improve (test)", {}).get("per_task", {})
    common = sorted(set(b.keys()) & set(ts.keys()))
    saved_ts, saved_si = 0, 0
    print(f"{'Task':>6s}  {'Base':>5s}  {'T→S':>5s}  {'Self':>5s}  {'Δ T→S':>6s}  {'Δ Self':>6s}")
    for tid in common:
        d_ts = b[tid] - ts.get(tid, b[tid])
        d_si = b[tid] - si.get(tid, b[tid])
        saved_ts += d_ts
        saved_si += d_si
        s_val = si.get(tid, None)
        s_str = str(s_val) if s_val is not None else "-"
        d_si_str = f"{d_si:>+6d}" if s_val is not None else "     -"
        print(f"{tid:>6d}  {b[tid]:>5d}  {ts[tid]:>5d}  {s_str:>5s}  {d_ts:>+6d}  {d_si_str}")
    print(f"{'TOTAL':>6s}  {'':>5s}  {'':>5s}  {'':>5s}  {saved_ts:>+6d}  {saved_si:>+6d}")

json.dump(results, open(os.path.join(OUT, "s7_step_efficiency.json"), "w"), indent=2)
print(f"\nSaved to {OUT}/s7_step_efficiency.json")
