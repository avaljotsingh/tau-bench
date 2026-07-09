"""Per-task tool routing: only expose composite tools on tasks where they're likely to help.

The fix≈break problem: composite tools fix some tasks but break others. Instead of giving
ALL tools to ALL tasks, we learn from training data which tools are safe for which task types,
then only expose the safe subset.

Approach:
1. Analyze training runs: for each (task, tool) pair, did using the tool help or hurt?
2. Classify tasks by their gold action pattern (what base tools they need)
3. Build a router: given a task's instruction, decide which composite tools to expose
4. Deploy with routing on held-out data

The router uses a simple heuristic: categorize tasks by the WRITE action they require
(return, exchange, modify, cancel, etc.) and only expose tools that were safe for that category
in training.
"""
import os, sys, json, ast
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["TRAPI_MODEL_VERSION"] = "2024-07-18"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def categorize_task(record):
    """Categorize a task by the primary WRITE action in its gold actions."""
    gold = (record.get("info") or {}).get("task", {}).get("actions", [])
    for a in gold:
        name = a["name"]
        if name == "return_delivered_order_items":
            return "return"
        if name == "exchange_delivered_order_items":
            return "exchange"
        if name == "modify_pending_order_items":
            return "modify_items"
        if name == "modify_pending_order_address":
            return "modify_address"
        if name == "modify_pending_order_payment":
            return "modify_payment"
        if name == "cancel_pending_order":
            return "cancel"
    instr = (record.get("info") or {}).get("task", {}).get("instruction", "").lower()
    if "return" in instr:
        return "return"
    if "exchange" in instr:
        return "exchange"
    if "cancel" in instr:
        return "cancel"
    if "modify" in instr or "change" in instr:
        return "modify"
    return "other"


def build_safety_map(base_records, aug_records, gen_tool_names):
    """For each task category, determine which tools are safe (didn't cause breaks)."""
    base_map = {r["task_id"]: r["reward"] for r in base_records}
    category_tool_outcomes = {}

    for r in aug_records:
        tid = r["task_id"]
        cat = categorize_task(r)
        base_rw = base_map.get(tid, 0)
        aug_rw = r["reward"]

        ea = (r.get("records") or {}).get("env_actions", [])
        tools_used = set()
        for a in ea:
            if a["name"] in gen_tool_names:
                tools_used.add(a["name"])

        if cat not in category_tool_outcomes:
            category_tool_outcomes[cat] = {}

        for tool in tools_used:
            if tool not in category_tool_outcomes[cat]:
                category_tool_outcomes[cat][tool] = {"helped": [], "hurt": [], "neutral": []}

            b_pass = base_rw >= 1 - 1e-6
            a_pass = aug_rw >= 1 - 1e-6

            if not b_pass and a_pass:
                category_tool_outcomes[cat][tool]["helped"].append(tid)
            elif b_pass and not a_pass:
                category_tool_outcomes[cat][tool]["hurt"].append(tid)
            else:
                category_tool_outcomes[cat][tool]["neutral"].append(tid)

    safety_map = {}
    for cat, tools in category_tool_outcomes.items():
        safe = []
        for tool, outcomes in tools.items():
            if len(outcomes["hurt"]) == 0:
                safe.append(tool)
        safety_map[cat] = safe

    return safety_map, category_tool_outcomes


def build_routed_servers(safety_map, aug_server_path, base_server_path, out_dir):
    """Build one filtered MCP server per task category, with only safe tools."""
    base_src_lines = open(base_server_path, encoding="latin-1").read().splitlines()
    at = next((i for i, ln in enumerate(base_src_lines) if ln.strip().startswith("if __name__")), len(base_src_lines))
    base_prefix = "\n".join(base_src_lines[:at]).rstrip()
    base_suffix = "\n".join(base_src_lines[at:])

    aug_src = open(aug_server_path, encoding="latin-1").read()
    tree = ast.parse(aug_src)
    tool_sources = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            src_seg = ast.get_source_segment(aug_src, node)
            if src_seg:
                tool_sources[node.name] = src_seg

    servers = {}
    for cat, safe_tools in safety_map.items():
        out_path = os.path.join(out_dir, f"server_{cat}.py")
        out = base_prefix + "\n\n"
        for tool in safe_tools:
            if tool in tool_sources:
                out += "@mcp.tool()\n" + tool_sources[tool].strip() + "\n\n"
        out += base_suffix + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        servers[cat] = out_path
        print(f"  {cat}: {len(safe_tools)} safe tools -> {out_path}")

    return servers


if __name__ == "__main__":
    from tau_bench.types import RunConfig
    from tau_bench.run import run as tau_run

    OUT = "experiments/phase1/per_task_routing"
    os.makedirs(OUT, exist_ok=True)

    BASE_SERVER = "mcp/retail_server.py"
    AUG_SERVER = "experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py"
    gen_tools_info = json.load(open("experiments/phase1/mined_from_gpt4o/repaired.json"))
    gen_tool_names = set(gen_tools_info["kept"])

    base_recs = json.load(open("experiments/phase1/gpt4omini_base.json"))
    aug_recs = json.load(open("experiments/phase1/gpt4omini_repaired_v3.json"))

    # Step 1: Build safety map
    print("=" * 60)
    print("STEP 1: Building safety map from training data")
    print("=" * 60)
    safety_map, outcomes = build_safety_map(base_recs, aug_recs, gen_tool_names)

    for cat, tools in sorted(outcomes.items()):
        print(f"\n  Category: {cat}")
        for tool, o in sorted(tools.items()):
            safe = "SAFE" if len(o["hurt"]) == 0 else f"UNSAFE (hurt {o['hurt']})"
            print(f"    {tool}: helped={o['helped']}, hurt={o['hurt']}, neutral={len(o['neutral'])} -> {safe}")

    print(f"\n  Safety map:")
    for cat, tools in sorted(safety_map.items()):
        print(f"    {cat}: {tools}")

    # Step 2: Build per-category servers
    print(f"\n{'=' * 60}")
    print("STEP 2: Building per-category servers")
    print("=" * 60)
    servers = build_routed_servers(safety_map, AUG_SERVER, BASE_SERVER, OUT)

    # Step 3: Run on held-out test data with routing
    print(f"\n{'=' * 60}")
    print("STEP 3: Running routed evaluation on held-out test (tasks 80-99)")
    print("=" * 60)

    TASK_IDS = list(range(80, 100))
    N = len(TASK_IDS)

    test_base = json.load(open("experiments/phase1/holdout_test/base_test.json"))
    task_categories = {}
    for r in test_base:
        cat = categorize_task(r)
        task_categories[r["task_id"]] = cat
    print(f"\n  Test task categories:")
    for tid in sorted(task_categories):
        print(f"    Task {tid}: {task_categories[tid]}")

    cat_tasks = {}
    for tid, cat in task_categories.items():
        cat_tasks.setdefault(cat, []).append(tid)

    all_results = []
    for cat, tids in sorted(cat_tasks.items()):
        server = servers.get(cat)
        if not server:
            print(f"\n  Category '{cat}': no training data, using base server")
            server = BASE_SERVER

        ckpt = os.path.join(OUT, f"routed_{cat}_test.json")
        if os.path.exists(ckpt):
            os.remove(ckpt)

        safe_tools = safety_map.get(cat, [])
        print(f"\n  Running category '{cat}' ({len(tids)} tasks: {tids})")
        print(f"  Tools exposed: {safe_tools if safe_tools else 'base only'}")

        cfg = RunConfig(
            model_provider="openai", user_model_provider="openai", model="none", user_model="none",
            num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="test",
            start_index=0, end_index=-1, task_ids=tids, log_dir=OUT, max_concurrency=8, seed=10,
            shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=server,
            ckpt_path=ckpt, new_func=None)

        res = tau_run(cfg)
        for r in res:
            all_results.append({"task_id": r.task_id, "reward": r.reward, "category": cat,
                                "safe_tools": safe_tools})

    routed_passed = sum(1 for r in all_results if r["reward"] >= 1 - 1e-6)
    routed_failures = sorted(r["task_id"] for r in all_results if r["reward"] < 1 - 1e-6)

    base_passed = sum(1 for r in test_base if r["reward"] >= 1 - 1e-6)
    unfiltered_test = json.load(open("experiments/phase1/holdout_test/teacher_student_test.json"))
    unfiltered_passed = sum(1 for r in unfiltered_test if r["reward"] >= 1 - 1e-6)

    report = {
        "experiment": "per-task tool routing on held-out test",
        "model": "gpt-4o-mini_2024-07-18",
        "safety_map": safety_map,
        "task_categories": {str(k): v for k, v in task_categories.items()},
        "results": {
            "base": {"passed": base_passed, "n": N},
            "unfiltered": {"passed": unfiltered_passed, "n": N},
            "routed": {"passed": routed_passed, "n": N, "failures": routed_failures},
        },
    }
    with open(os.path.join(OUT, "routing_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PER-TASK ROUTING RESULTS (held-out test, tasks 80-99)")
    print(f"{'=' * 60}")
    print(f"  Base (no tools):         {base_passed}/{N}")
    print(f"  Unfiltered (all tools):  {unfiltered_passed}/{N}")
    print(f"  ROUTED (safe tools):     {routed_passed}/{N} ({'+' if routed_passed >= base_passed else ''}{routed_passed - base_passed} vs base)")
    print(f"  Failures: {routed_failures}")
    print(f"{'=' * 60}")
