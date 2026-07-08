"""Score a generated tool by the original idea: replay a trajectory that uses it, see if it passes.

Deterministic — no live agent. Writes results to experiments/scoring/offline/ (kept separate from any
live/stochastic results).

For each task in a run file we replay the recorded action sequence through the offline harness and compute
reward = (final DB hash == gold DB hash). We report the deterministic pass/fail and compare it to the
recorded reward.

Usage: python scoring/score_tool.py <run_json> [label]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring import replay as R

SERVER = "experiments/weak_experiment/augmented.py"   # has base + all generated tools
OUT_DIR = "experiments/scoring/offline"


def score_run(run_json, label=None):
    label = label or os.path.splitext(os.path.basename(run_json))[0]
    funcs = R.load_server_funcs(SERVER)
    recs = json.load(open(run_json))
    per_task, agree = {}, 0
    for rec in recs:
        gold = R.actions_from_gold(rec["info"])
        agent = R.actions_from_record(rec)
        gt = R.gold_hash(gold, funcs)
        rw = R.reward_of(R.replay(agent, funcs), gt)
        rec_rw = 1.0 if rec["reward"] >= 1 - 1e-6 else 0.0
        used = [n for n, _ in agent]
        per_task[rec["task_id"]] = {"replay_reward": rw, "recorded_reward": rec_rw,
                                     "n_calls": len(agent)}
        agree += (rw == rec_rw)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{label}.json")
    passed = sum(1 for v in per_task.values() if v["replay_reward"] == 1.0)
    report = {"label": label, "run": run_json, "n_tasks": len(per_task),
              "deterministic_passed": passed,
              "agreement_with_recorded": f"{agree}/{len(per_task)}",
              "per_task": {str(k): per_task[k] for k in sorted(per_task)}}
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{label}] deterministic pass={passed}/{len(per_task)}  "
          f"agreement-with-recorded={agree}/{len(per_task)}  -> {out}")
    return report


if __name__ == "__main__":
    run_json = sys.argv[1] if len(sys.argv) > 1 else "experiments/weak_experiment/augmented_run.json"
    label = sys.argv[2] if len(sys.argv) > 2 else None
    score_run(run_json, label)
