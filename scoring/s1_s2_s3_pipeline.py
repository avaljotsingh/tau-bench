"""S1–S3 Deterministic Scoring Pipeline for generated composite tools.

For each candidate tool, tests whether it preserves correctness on passing tasks (S3)
and whether it can fix failing tasks (S2), all via deterministic offline replay.

The key insight: reward = data_hash == gt_data_hash is a DETERMINISTIC function of the
final DB state. We can score any tool by replaying gold action sequences with/without it,
completely offline, no live agent, no stochastic user, no noise.

Pipeline per tool:
  S1 - Substitution correctness: replay gold actions through augmented server, verify reward preserved
  S2 - Counterfactual fix: for failing tasks, does inserting the tool flip 0→1?
  S3 - Deterministic regression: for ALL passing tasks, does adding the tool cause ANY reward drops?

Selection rule: KEEP tool iff S3 regressions == 0 (never break a passing task)
"""
import os, sys, json, ast, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring import replay as R
from typing import Dict, List, Tuple

BASE_SERVER = "mcp/retail_server.py"
OUT_DIR = "experiments/scoring/offline"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_tool_names_from_server(server_path: str) -> List[str]:
    """Get function names defined in a server file."""
    src = open(server_path, encoding="latin-1").read()
    return [n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)]


def get_base_tool_names() -> set:
    """Names of tools in the base server (not generated)."""
    return set(extract_tool_names_from_server(BASE_SERVER))


def score_tool_s1_s3(
    tool_name: str,
    augmented_server: str,
    records: List[dict],
    verbose: bool = True,
) -> dict:
    """S1 + S3: replay gold actions through base vs augmented server on PASSING tasks.

    S1: Does the augmented server preserve reward on all tasks? (substitution correctness)
    S3: Does adding the tool cause ANY regressions? (zero tolerance)

    We compare base_reward vs augmented_reward for each passing task.
    A regression = base passes but augmented fails.
    """
    base_funcs = R.load_server_funcs(BASE_SERVER)
    aug_funcs = R.load_server_funcs(augmented_server)

    passing = [r for r in records if r["reward"] >= 1 - 1e-6]
    results_per_task = {}
    regressions = []
    preserved = 0

    for rec in passing:
        tid = rec["task_id"]
        gold = R.actions_from_gold(rec["info"])
        if not gold:
            continue

        # Replay with base server
        base_final = R.replay(gold, base_funcs)
        base_gt = R.data_hash(base_final)

        # Replay with augmented server (same gold actions)
        aug_final = R.replay(gold, aug_funcs)
        aug_hash = R.data_hash(aug_final)

        base_reward = 1.0  # gold actions on base always pass by construction
        aug_reward = 1.0 if aug_hash == base_gt else 0.0

        results_per_task[tid] = {
            "base_reward": base_reward,
            "aug_reward": aug_reward,
            "regressed": aug_reward < base_reward,
        }

        if aug_reward < base_reward:
            regressions.append(tid)
            if verbose:
                print(f"  [S3 REGRESSION] task {tid}: base=PASS, aug=FAIL")
        else:
            preserved += 1

    n_tested = len(results_per_task)
    result = {
        "tool": tool_name,
        "gate": "S1+S3",
        "n_passing_tested": n_tested,
        "preserved": preserved,
        "regressions": regressions,
        "n_regressions": len(regressions),
        "S3_pass": len(regressions) == 0,
        "per_task": {str(k): v for k, v in sorted(results_per_task.items())},
    }

    if verbose:
        status = "PASS" if result["S3_pass"] else "FAIL"
        print(f"  [S1+S3] {tool_name}: {preserved}/{n_tested} preserved, "
              f"{len(regressions)} regressions -> {status}")

    return result


def score_tool_s2(
    tool_name: str,
    augmented_server: str,
    records: List[dict],
    verbose: bool = True,
) -> dict:
    """S2: Counterfactual fix — for failing tasks, does the augmented server fix any?

    Replay the agent's ACTUAL actions (which failed) through the augmented server.
    If the augmented server has the tool, and the agent happened to call tools that
    the composite wraps, the replay might produce a different (correct) outcome.

    Also replay gold actions through augmented server on failing tasks —
    if gold actions pass with augmented server, the tool at least doesn't interfere.
    """
    base_funcs = R.load_server_funcs(BASE_SERVER)
    aug_funcs = R.load_server_funcs(augmented_server)

    failing = [r for r in records if r["reward"] < 1 - 1e-6]
    results_per_task = {}
    fixes = []

    for rec in failing:
        tid = rec["task_id"]
        gold = R.actions_from_gold(rec["info"])
        agent_actions = R.actions_from_record(rec)

        if not gold:
            continue

        # Gold hash (ground truth)
        base_gt_final = R.replay(gold, base_funcs)
        gt_hash = R.data_hash(base_gt_final)

        # Replay agent's actual (failing) actions with augmented server
        aug_agent_final = R.replay(agent_actions, aug_funcs)
        aug_agent_reward = R.reward_of(aug_agent_final, gt_hash)

        # Also check: do gold actions still pass with augmented server?
        aug_gold_final = R.replay(gold, aug_funcs)
        aug_gold_reward = R.reward_of(aug_gold_final, gt_hash)

        fixed = aug_agent_reward >= 1 - 1e-6
        results_per_task[tid] = {
            "agent_replay_reward": aug_agent_reward,
            "gold_replay_reward": aug_gold_reward,
            "fixed": fixed,
        }

        if fixed:
            fixes.append(tid)
            if verbose:
                print(f"  [S2 FIX] task {tid}: agent replay now PASSES with augmented server!")

    result = {
        "tool": tool_name,
        "gate": "S2",
        "n_failing_tested": len(results_per_task),
        "fixes": fixes,
        "n_fixes": len(fixes),
        "per_task": {str(k): v for k, v in sorted(results_per_task.items())},
    }

    if verbose:
        print(f"  [S2] {tool_name}: {len(fixes)}/{len(results_per_task)} failing tasks fixed")

    return result


def score_library(
    label: str,
    augmented_server: str,
    tool_names: List[str],
    records: List[dict],
    verbose: bool = True,
) -> dict:
    """Score an entire augmented library (all tools together) through S1-S3.

    This tests the LIBRARY as a unit — do the tools collectively cause regressions?
    Individual tool attribution would require per-tool augmented servers.
    """
    print(f"\n{'='*60}")
    print(f"Scoring library: {label}")
    print(f"Tools: {tool_names}")
    print(f"Server: {augmented_server}")
    print(f"{'='*60}")

    s1_s3 = score_tool_s1_s3(label, augmented_server, records, verbose)
    s2 = score_tool_s2(label, augmented_server, records, verbose)

    report = {
        "label": label,
        "server": augmented_server,
        "tools": tool_names,
        "S1_S3": s1_s3,
        "S2": s2,
        "summary": {
            "S3_pass": s1_s3["S3_pass"],
            "regressions": s1_s3["regressions"],
            "fixes": s2["fixes"],
            "net_benefit": s2["n_fixes"] - s1_s3["n_regressions"],
            "keep": s1_s3["S3_pass"],  # strict: zero regressions
        }
    }

    out_path = os.path.join(OUT_DIR, f"{label}_s1s2s3.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- {label} SUMMARY ---")
    print(f"  S3 (regression gate): {'PASS' if s1_s3['S3_pass'] else 'FAIL'} "
          f"({s1_s3['n_regressions']} regressions)")
    print(f"  S2 (fixes):           {s2['n_fixes']} failing tasks fixed")
    print(f"  Net benefit:          {s2['n_fixes'] - s1_s3['n_regressions']}")
    print(f"  KEEP:                 {'YES' if report['summary']['keep'] else 'NO'}")
    print(f"  Saved to: {out_path}")

    return report


def score_individual_tools(
    label: str,
    augmented_server: str,
    tool_names: List[str],
    records: List[dict],
    base_tool_names: set,
    verbose: bool = True,
) -> List[dict]:
    """Score each generated tool INDIVIDUALLY by building a per-tool augmented server.

    For each tool, creates a server with base + ONLY that tool, then runs S1-S3.
    This gives per-tool attribution — which specific tools cause regressions.
    """
    from scoring import verify_repair as V

    base_src_lines = open(BASE_SERVER, encoding="latin-1").read().splitlines()
    at = next((i for i, ln in enumerate(base_src_lines) if ln.strip().startswith("if __name__")), len(base_src_lines))
    base_prefix = "\n".join(base_src_lines[:at]).rstrip()
    base_suffix = "\n".join(base_src_lines[at:])

    aug_src = open(augmented_server, encoding="latin-1").read()
    tree = ast.parse(aug_src)
    tool_sources = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in tool_names:
            tool_sources[node.name] = ast.get_source_segment(aug_src, node)

    tmp_server = os.path.join(OUT_DIR, "_tmp_single_tool_server.py")
    individual_results = []

    for name in tool_names:
        if name not in tool_sources:
            print(f"  WARNING: {name} not found in server source, skipping")
            continue

        # Build server with base + this one tool
        single_aug = base_prefix + "\n\n@mcp.tool()\n" + tool_sources[name].strip() + "\n\n" + base_suffix + "\n"
        with open(tmp_server, "w", encoding="utf-8") as f:
            f.write(single_aug)

        print(f"\n--- Scoring individual tool: {name} ---")
        s1_s3 = score_tool_s1_s3(name, tmp_server, records, verbose)
        s2 = score_tool_s2(name, tmp_server, records, verbose)

        result = {
            "tool": name,
            "S1_S3": s1_s3,
            "S2": s2,
            "S3_pass": s1_s3["S3_pass"],
            "regressions": s1_s3["regressions"],
            "fixes": s2["fixes"],
            "net": s2["n_fixes"] - s1_s3["n_regressions"],
            "keep": s1_s3["S3_pass"],
        }
        individual_results.append(result)

    # Clean up
    if os.path.exists(tmp_server):
        os.remove(tmp_server)

    return individual_results


def build_filtered_server(
    survivors: List[str],
    augmented_server: str,
    output_path: str,
):
    """Build a new server with base + only the tools that passed S3."""
    base_src_lines = open(BASE_SERVER, encoding="latin-1").read().splitlines()
    at = next((i for i, ln in enumerate(base_src_lines) if ln.strip().startswith("if __name__")), len(base_src_lines))

    aug_src = open(augmented_server, encoding="latin-1").read()
    tree = ast.parse(aug_src)

    out = "\n".join(base_src_lines[:at]).rstrip() + "\n\n"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in survivors:
            src = ast.get_source_segment(aug_src, node)
            out += "@mcp.tool()\n" + src.strip() + "\n\n"
    out += "\n".join(base_src_lines[at:]) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Built filtered server with {len(survivors)} tools -> {output_path}")


if __name__ == "__main__":
    records = json.load(open("experiments/phase1/gpt4omini_base.json"))
    base_tools = get_base_tool_names()

    # Score both libraries
    libraries = [
        {
            "label": "teacher_student",
            "server": "experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py",
            "tools": json.load(open("experiments/phase1/mined_from_gpt4o/repaired.json"))["kept"],
        },
        {
            "label": "self_improve",
            "server": "experiments/phase1/self_improve/augmented_repaired.py",
            "tools": json.load(open("experiments/phase1/self_improve/repaired.json"))["kept"],
        },
    ]

    all_survivors = {}

    for lib in libraries:
        # Score the whole library
        lib_report = score_library(
            lib["label"], lib["server"], lib["tools"], records
        )

        # Score each tool individually for attribution
        print(f"\n{'='*60}")
        print(f"Individual tool scoring: {lib['label']}")
        print(f"{'='*60}")
        individual = score_individual_tools(
            lib["label"], lib["server"], lib["tools"], records, base_tools
        )

        # Save individual results
        ind_path = os.path.join(OUT_DIR, f"{lib['label']}_individual.json")
        with open(ind_path, "w") as f:
            json.dump(individual, f, indent=2)

        # Identify survivors (tools with zero regressions)
        survivors = [r["tool"] for r in individual if r["keep"]]
        dropped = [r["tool"] for r in individual if not r["keep"]]
        all_survivors[lib["label"]] = {
            "survivors": survivors,
            "dropped": dropped,
            "individual": individual,
        }

        print(f"\n{'='*60}")
        print(f"{lib['label']} INDIVIDUAL RESULTS:")
        print(f"  Survivors (S3 pass): {survivors}")
        print(f"  Dropped (S3 fail):   {dropped}")
        print(f"{'='*60}")

        # Build filtered server with only survivors
        if survivors:
            filtered_path = os.path.join(OUT_DIR, f"{lib['label']}_filtered_server.py")
            build_filtered_server(survivors, lib["server"], filtered_path)

    # Final summary
    summary = {
        "experiment": "S1-S3 deterministic scoring pipeline",
        "description": "Per-tool offline scoring: S3=zero regressions on passing tasks, S2=fixes on failing tasks",
        "libraries": {},
    }
    for label, data in all_survivors.items():
        summary["libraries"][label] = {
            "survivors": data["survivors"],
            "dropped": data["dropped"],
            "per_tool": [
                {
                    "tool": r["tool"],
                    "S3_pass": r["S3_pass"],
                    "regressions": r["regressions"],
                    "fixes": r["fixes"],
                    "net": r["net"],
                }
                for r in data["individual"]
            ],
        }

    with open(os.path.join(OUT_DIR, "s1s2s3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL S1-S3 SUMMARY")
    print(f"{'='*60}")
    for label, data in all_survivors.items():
        print(f"\n{label}:")
        print(f"  Survivors: {data['survivors']}")
        print(f"  Dropped:   {data['dropped']}")
        for r in data["individual"]:
            flag = "✓" if r["keep"] else "✗"
            print(f"    {flag} {r['tool']}: regressions={r['regressions']}, fixes={r['fixes']}, net={r['net']}")
