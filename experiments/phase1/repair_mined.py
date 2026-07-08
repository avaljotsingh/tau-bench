"""Run verify->repair on the mined tools and rebuild a repaired library."""
import os, sys, ast, json
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"
os.environ.setdefault("LIBGEN_GEN_DEPLOYMENT", "o4-mini_2025-04-16")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scoring import verify_repair as V

BASE_SERVER = "mcp/retail_server.py"
AUG = "experiments/phase1/mined_from_gpt4o/augmented_mined.py"
OUT = "experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py"

names = json.load(open("experiments/phase1/mined_from_gpt4o/mined.json"))["tools"]
src = open(AUG, encoding="latin-1").read()
tool_src = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name in names}

# real argument values to test tools on
records = json.load(open("experiments/phase1/gpt4omini_base.json"))
pool = V.build_arg_pool(records)

kept, dropped = {}, []
for name in names:
    fixed, ok, attempts = V.verify_and_repair(name, tool_src[name], BASE_SERVER, pool, max_retries=5)
    if ok:
        kept[name] = fixed
    else:
        dropped.append(name)

# rebuild repaired library
b = open(BASE_SERVER, encoding="latin-1").read().splitlines()
at = next((i for i, ln in enumerate(b) if ln.strip().startswith("if __name__")), len(b))
out = "\n".join(b[:at]).rstrip() + "\n\n"
for s in kept.values():
    out += "@mcp.tool()\n" + s.strip() + "\n\n"
out += "\n".join(b[at:]) + "\n"
open(OUT, "w", encoding="utf-8").write(out)
json.dump({"kept": list(kept), "dropped": dropped}, open("experiments/phase1/mined_from_gpt4o/repaired.json", "w"), indent=2)
print(f"\n=== verify+repair done: kept {len(kept)} ({list(kept)}), dropped {dropped} ===")
print(f"repaired library -> {OUT}")
