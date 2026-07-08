"""Weak-agent failure-driven experiment (parallelized).

Rationale: gpt-5 saturates retail accuracy (no headroom) and barely fails (n=5).
A WEAK agent (gpt-4o-mini) fails a lot -> a real failure set to learn from AND real
headroom for composite tools to help.

Flow (all task runs parallel via tau_bench's executor; tool generation parallel too):
  1. Base run (weak agent) on N tasks -> base accuracy + failures.
  2. Generate failure-targeted tools from the failures IN PARALLEL (deterministic synth).
  3. Build augmented library (base + tools).
  4. Augmented run (weak agent + nudge) on the SAME N tasks.
  5. Report base vs augmented: accuracy, steps, fixes (fail->pass), regressions (pass->fail).

Model is taken from TRAPI_* env (set gpt-4o-mini before running).
Usage: python experiments/run_weak_experiment.py [n_tasks] [gen_cap]
"""
import os, sys, json, ast
from concurrent.futures import ThreadPoolExecutor

# Model defaults to gpt-4o-mini (weak). Override via env before launching.
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-07-18")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-4o-mini_2024-07-18")

import lib_gen
from libgen_utils import get_tools, Library
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
GEN_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TASK_IDS = list(range(0, N))
OUT_DIR = "experiments/weak_experiment"
os.makedirs(OUT_DIR, exist_ok=True)
BASE_SERVER = "mcp/retail_server.py"
AUG_SERVER = os.path.join(OUT_DIR, "augmented.py")
NUDGE = (" IMPORTANT: Higher-level composite tools may be available that bundle a multi-step "
         "lookup or enforce a step agents often get wrong. When one fits, prefer a single call to it "
         "instead of the lower-level calls, and don't redo the same work afterward.")

def cfg(server, ckpt):
    return RunConfig(model_provider="openai", user_model_provider="openai", model="none", user_model="none",
        num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.2, task_split="test",
        start_index=0, end_index=-1, task_ids=TASK_IDS, log_dir=OUT_DIR, max_concurrency=8, seed=10,
        shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=server, ckpt_path=ckpt, new_func=None)

def tool_calls(t): return sum(1 for m in t if isinstance(m, dict) and m.get("role") == "tool")

def fail_reason(r):
    task = r.get("info", {}).get("task", {})
    actual = [{"name": (tc or {}).get("function", {}).get("name"), "args": (tc or {}).get("function", {}).get("arguments")}
              for m in r.get("traj", []) if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
              for tc in m["tool_calls"]]
    return (f"The agent FAILED (reward={r.get('reward')}). Instruction: {task.get('instruction')}\n"
            f"Expected ground-truth actions: {json.dumps(task.get('actions', []))}\n"
            f"Agent's actual tool calls: {json.dumps(actual)}\n"
            f"Required outputs: {task.get('outputs', [])}")

def assemble(tool_sources):
    src = open(BASE_SERVER, encoding="latin-1").read().splitlines()
    at = next((i for i, ln in enumerate(src) if ln.strip().startswith("if __name__")), len(src))
    head = "\n".join(src[:at]).rstrip() + "\n"; tail = "\n".join(src[at:])
    blocks = ["\n@mcp.tool()\n" + s.strip() + "\n" for s in tool_sources]
    out = head + "\n" + "\n".join(blocks) + "\n\n" + tail + "\n"
    ast.parse(out)
    open(AUG_SERVER, "w", encoding="utf-8").write(out)

def acc(results): return sum(1 for r in results if r.reward >= 1 - 1e-6)

print(f"=== ARM A: BASE ({os.environ['TRAPI_MODEL_NAME']}) on {N} tasks ===", flush=True)
os.environ["AGENT_PROMPT_SUFFIX"] = ""
base_res = tau_run(cfg(BASE_SERVER, os.path.join(OUT_DIR, "base.json")))
base_acc = acc(base_res)
fails = [r for r in base_res if r.reward < 1 - 1e-6]
print(f"BASE accuracy: {base_acc}/{N}; {len(fails)} failures", flush=True)

print(f"=== GENERATE tools from {min(len(fails), GEN_CAP)} failures (parallel) ===", flush=True)
base_lib = Library(get_tools(BASE_SERVER)).get_funcs()
def gen(r):
    name, s = lib_gen.get_new_func_from_failure(r.traj, fail_reason({"reward": r.reward, "info": r.info, "traj": r.traj}), base_lib, verbose=False)
    return r.task_id, name, s
tools = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    for tid, name, s in ex.map(gen, fails[:GEN_CAP]):
        if s and name not in tools:
            tools[name] = s; print(f"  [task {tid}] tool: {name}", flush=True)
assemble(list(tools.values()))
print(f"Augmented library: base + {len(tools)} tools: {list(tools)}", flush=True)

print(f"=== ARM B: AUGMENTED + nudge on the SAME {N} tasks ===", flush=True)
os.environ["AGENT_PROMPT_SUFFIX"] = NUDGE
aug_res = tau_run(cfg(AUG_SERVER, os.path.join(OUT_DIR, "augmented_run.json")))
os.environ["AGENT_PROMPT_SUFFIX"] = ""
aug_acc = acc(aug_res)

bmap = {r.task_id: r.reward for r in base_res}; amap = {r.task_id: r.reward for r in aug_res}
fixed = [t for t in bmap if bmap[t] < 1 - 1e-6 and amap.get(t, 0) >= 1 - 1e-6]
broke = [t for t in bmap if bmap[t] >= 1 - 1e-6 and amap.get(t, 0) < 1 - 1e-6]
report = {"model": os.environ["TRAPI_MODEL_NAME"], "n_tasks": N,
    "base_accuracy": base_acc, "augmented_accuracy": aug_acc, "net_delta": aug_acc - base_acc,
    "base_failures": len(fails), "tools_generated": list(tools),
    "fixed_fail_to_pass": sorted(fixed), "broke_pass_to_fail": sorted(broke),
    "base_avg_tool_calls": round(sum(tool_calls(r.traj) for r in base_res) / N, 2),
    "augmented_avg_tool_calls": round(sum(tool_calls(r.traj) for r in aug_res) / N, 2)}
json.dump(report, open(os.path.join(OUT_DIR, "report.json"), "w"), indent=2)
print("\n================ WEAK-AGENT RESULT ================")
print(json.dumps(report, indent=2))
