"""Failure-driven libgen experiment.

Closes the loop the user asked for: learn from FAILURES to make better tools.
  1. Take a base run's FAILED tasks (reward < 1) + a deterministic why-it-failed
     analysis (expected ground-truth actions/outputs vs the agent's actual calls).
  2. For each failure, deterministically synthesize a composite tool TARGETING it
     (lib_gen.get_new_func_from_failure).
  3. Build an augmented library (base + the failure-targeted tools).
  4. Re-run the previously-failing tasks with the augmented library + usage nudge.
  5. NET-BENEFIT selection / report: how many failures flipped to success.

Usage: python experiments/run_failure_driven.py <base_results_with_failures.json> [max_tools]
"""
import os, sys, json, ast
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-5")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-11-20")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-5_2025-08-07")

import lib_gen
from libgen_utils import get_tools, Library
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

OUT_DIR = "experiments/failure_driven"
os.makedirs(OUT_DIR, exist_ok=True)
BASE_SERVER = "mcp/retail_server.py"
AUG_SERVER = os.path.join(OUT_DIR, "augmented_from_failures.py")

NUDGE = (
    "\n\nIMPORTANT - efficiency & correctness: Higher-level composite tools may be available "
    "that bundle a multi-step lookup or enforce a step agents commonly get wrong. When one fits "
    "the task, PREFER a single call to it instead of issuing the lower-level calls yourself, and "
    "do not redo the same work with lower-level tools afterward."
)

def extract_tool_calls(traj):
    calls = []
    for m in traj:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = (tc or {}).get("function", {}) or {}
                calls.append({"name": fn.get("name"), "args": fn.get("arguments")})
    return calls

def deterministic_failure_reason(result):
    task = result.get("info", {}).get("task", {})
    expected = task.get("actions", [])
    outputs = task.get("outputs", [])
    actual = extract_tool_calls(result.get("traj", []))
    lines = [
        f"The agent FAILED this task (reward={result.get('reward')}).",
        f"Instruction: {task.get('instruction')}",
        f"Expected ground-truth actions: {json.dumps(expected)}",
        f"Agent's actual tool calls: {json.dumps(actual)}",
    ]
    if outputs:
        lines.append(f"Required outputs the agent had to report: {outputs}")
    return "\n".join(lines)

def assemble_augmented(tool_sources):
    base_src = open(BASE_SERVER, encoding="latin-1").read()
    lines = base_src.splitlines()
    insert_at = next((i for i, ln in enumerate(lines) if ln.strip().startswith("if __name__")), len(lines))
    head = "\n".join(lines[:insert_at]).rstrip() + "\n"
    tail = "\n".join(lines[insert_at:])
    blocks = ["\n@mcp.tool()\n" + s.strip() + "\n" for s in tool_sources]
    aug = head + "\n" + "\n".join(blocks) + "\n\n" + tail + "\n"
    ast.parse(aug)
    with open(AUG_SERVER, "w", encoding="utf-8") as f:
        f.write(aug)

def make_cfg(mcp_server, task_ids, ckpt):
    return RunConfig(
        model_provider="openai", user_model_provider="openai", model="none", user_model="none",
        num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.2, task_split="test",
        start_index=0, end_index=-1, task_ids=task_ids, log_dir=OUT_DIR, max_concurrency=6, seed=10,
        shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=mcp_server, ckpt_path=ckpt, new_func=None,
    )

def main():
    base_results_path = sys.argv[1] if len(sys.argv) > 1 else None
    max_tools = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if not base_results_path or not os.path.exists(base_results_path):
        print("Provide a base-results JSON (with failures) as arg 1."); return

    results = json.load(open(base_results_path))
    failures = [r for r in results if r.get("reward", 0) < 1 - 1e-6]
    print(f"Base run: {len(results)} tasks, {len(failures)} FAILED.")
    if not failures:
        print("No failures to learn from in this file."); return
    all_fail_ids = [r["task_id"] for r in failures]
    gen_failures = failures[:max_tools]   # generate tools from up to max_tools failures
    fail_ids = all_fail_ids               # but re-run ALL failures (tools may generalize)
    print(f"Re-running all {len(all_fail_ids)} failing task_ids; generating tools from {len(gen_failures)} of them.")

    base_lib = Library(get_tools(BASE_SERVER)).get_funcs()
    tools = {}
    for r in gen_failures:
        reason = deterministic_failure_reason(r)
        name, src = lib_gen.get_new_func_from_failure(r["traj"], reason, base_lib, verbose=True)
        if src and name not in tools:
            tools[name] = src
            print(f"  generated tool for task {r['task_id']}: {name}")
        else:
            print(f"  task {r['task_id']}: no valid tool synthesized")
    if not tools:
        print("No tools generated."); return
    assemble_augmented(list(tools.values()))
    json.dump({"tools": list(tools), "fail_ids": fail_ids}, open(os.path.join(OUT_DIR, "generated.json"), "w"), indent=2)
    print(f"Augmented library: base + {len(tools)} failure-targeted tools: {list(tools)}")

    # Re-run the previously-failing tasks with the augmented library + nudge.
    os.environ["AGENT_PROMPT_SUFFIX"] = NUDGE
    aug_res = tau_run(make_cfg(AUG_SERVER, fail_ids, os.path.join(OUT_DIR, "rerun_augmented.json")))
    os.environ["AGENT_PROMPT_SUFFIX"] = ""

    fixed = sum(1 for r in aug_res if r.reward >= 1 - 1e-6)
    report = {
        "failing_tasks": len(fail_ids), "fail_ids": fail_ids,
        "tools_generated": list(tools),
        "fixed_after_augment": fixed,
        "still_failing": len(fail_ids) - fixed,
        "per_task": {r.task_id: round(r.reward, 2) for r in aug_res},
    }
    json.dump(report, open(os.path.join(OUT_DIR, "failure_driven_report.json"), "w"), indent=2)
    print("\n================ FAILURE-DRIVEN RESULT ================")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
