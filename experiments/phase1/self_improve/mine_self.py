"""Self-improvement: mine composite tools from gpt-4o-mini's OWN solved trajectories.

Previous experiment mined from gpt-4o (teacher->student). This tests whether a model can
improve itself: mine from its own 39 solved trajectories, verify+repair, deploy back to itself.

Tool GENERATOR = o4-mini (best composer); TRAJECTORY SOURCE = gpt-4o-mini (same as deployment target).
"""
import os, sys, json
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"
os.environ.setdefault("LIBGEN_GEN_DEPLOYMENT", "o4-mini_2025-04-16")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lib_gen
from libgen_utils import Library, get_tools

BASE_SERVER = "mcp/retail_server.py"
OUT_DIR = "experiments/phase1/self_improve"
os.makedirs(OUT_DIR, exist_ok=True)
CHUNK = 8
N_TOOLS = 8
MAX_ATTEMPTS = 20

base_lib = Library(get_tools(BASE_SERVER)).get_funcs()

# Key difference: mine from gpt-4o-mini's OWN trajectories, not gpt-4o's
recs = json.load(open("experiments/phase1/gpt4omini_base.json"))
solved = [r for r in recs if r["reward"] >= 1 - 1e-6 and r.get("traj")]
print(f"gpt-4o-mini solved trajectories available: {len(solved)}")


def lib_with(mined):
    if isinstance(base_lib, dict):
        return {**base_lib, **mined}
    return list(base_lib) + list(mined.values())


tools = {}
attempt = 0
while len(tools) < N_TOOLS and attempt < MAX_ATTEMPTS:
    off = (attempt * 5) % max(1, len(solved) - CHUNK)
    chunk = solved[off: off + CHUNK]
    attempt += 1
    if not chunk:
        break
    print(f"\n--- attempt {attempt}: mining tool {len(tools)+1}/{N_TOOLS} from solved[{off}:{off+CHUNK}] ---", flush=True)
    try:
        name, src = lib_gen.get_new_func(chunk, lib_with(tools), verbose=True)
    except Exception as e:
        print("  mining failed:", repr(e)[:160]); continue
    if not src:
        print(f"  synthesis failed for '{name}' (invalid code) -> skip"); continue
    if name in tools or name in (base_lib if isinstance(base_lib, dict) else []):
        print(f"  '{name}' already in library -> skip (dupe)"); continue
    tools[name] = src
    print(f"  mined: {name}  ({len(tools)}/{N_TOOLS})")

# assemble augmented server = base + mined tools
base_src = open(BASE_SERVER, encoding="latin-1").read().splitlines()
at = next((j for j, ln in enumerate(base_src) if ln.strip().startswith("if __name__")), len(base_src))
aug = "\n".join(base_src[:at]).rstrip() + "\n\n"
for src in tools.values():
    aug += "@mcp.tool()\n" + src.strip() + "\n\n"
aug += "\n".join(base_src[at:]) + "\n"
open(os.path.join(OUT_DIR, "augmented_mined.py"), "w", encoding="utf-8").write(aug)
json.dump({"tools": list(tools), "source": "gpt-4o-mini solved trajectories (SELF)", "generator": os.environ["LIBGEN_GEN_DEPLOYMENT"]},
          open(os.path.join(OUT_DIR, "mined.json"), "w"), indent=2)
print(f"\n=== MINED {len(tools)} tools from gpt-4o-mini's OWN trajectories: {list(tools)} ===")
