"""Execute a generated tool; if it raises, feed the error back to the generator to REPAIR it (a couple
retries) before accepting or dropping it.

This closes the open-loop generation gap that produced the JSON-string bug (base tools return JSON strings,
not dicts, so the composite raised AttributeError on every call and trapped the agent in a retry loop).
"""
import os, json, re
from scoring import replay as R
from tau_bench.trapi_infer import gen_completion, model_dump

TMP_SERVER = "experiments/phase1/_verify_tmp_server.py"
CONTRACT = ("The base library tools (get_user_details, get_order_details, list_order_items, "
            "get_product_details, get_order_payment_details, etc.) RETURN JSON STRINGS, not dicts. "
            "Parse each result with json.loads() before using .get(). A tool may also return an error "
            "string beginning with 'Error:'; handle that before json.loads().")


def _build_server(base_server, tool_src, out_path):
    b = open(base_server, encoding="latin-1").read().splitlines()
    at = next((i for i, ln in enumerate(b) if ln.strip().startswith("if __name__")), len(b))
    open(out_path, "w", encoding="utf-8").write(
        "\n".join(b[:at]).rstrip() + "\n\n@mcp.tool()\n" + tool_src.strip() + "\n\n" + "\n".join(b[at:]) + "\n")


def _param_names(tool_src):
    m = re.search(r"def\s+\w+\s*\(([^)]*)\)", tool_src)
    if not m:
        return []
    return [p.split("=")[0].split(":")[0].strip() for p in m.group(1).split(",") if p.strip()]


def build_arg_pool(records):
    """Collect real argument values seen across trajectories, keyed by param name."""
    pool = {}
    for r in records:
        for _, kw in R.actions_from_record(r):
            for k, v in (kw or {}).items():
                if isinstance(v, (str, int)) and str(v):
                    pool.setdefault(k, [])
                    if v not in pool[k]:
                        pool[k].append(v)
    return pool


def _synth(param, pool):
    """Best-effort AGENT-STYLE value for a param: real IDs from the pool, else type-appropriate synthetic
    values (lists/numbers/dicts) matching how the LLM tool-caller would actually pass them."""
    p = param.lower()
    if param in pool and pool[param]:
        return pool[param][0]
    if any(k in p for k in ("price", "amount", "total", "cost")):
        return [100.0, 50.0] if p.endswith("s") else 100.0
    if p.endswith("_ids") or "item_ids" in p:
        return (pool.get("item_ids") or [["1008292230"]])[0]
    if "changes" in p or ("item" in p and p.endswith("s")):
        return []
    if "address" in p:
        return {"address1": "123 Main St", "city": "X", "state": "CA", "zip": "12345", "country": "USA"}
    if "payment" in p:
        return (pool.get("payment_method_id") or ["gift_card_0000000"])[0]
    if "type" in p or "action" in p:
        return "modify"
    if p.endswith("s"):
        return []
    return "test"


def make_test_args(tool_src, pool, k=3):
    params = _param_names(tool_src)
    if not params:
        return [{}]
    # agent-style inputs (never skip a tool now)
    return [{p: _synth(p, pool) for p in params} for _ in range(max(1, min(k, 2)))]


def execute_tool(base_server, tool_src, name, test_args_list):
    """Return (error_str, failing_args) on the first RAISED exception, else (None, None)."""
    _build_server(base_server, tool_src, TMP_SERVER)
    try:
        funcs = R.load_server_funcs(TMP_SERVER)
    except Exception as e:
        return f"load/compile error: {e!r}", None
    if name not in funcs:
        return f"tool '{name}' not defined after load", None
    R.seed_fresh()
    for args in test_args_list:
        try:
            out = funcs[name](**args)
        except Exception as e:
            return f"{type(e).__name__}: {e}", args
        # A tool that returns an error on plausible agent-style input is broken too (it makes the
        # agent retry-loop) even though it doesn't raise.
        if isinstance(out, dict) and "error" in out:
            return f"returned error on valid input: {out['error']}", args
        if isinstance(out, str) and out.strip().lower().startswith("error"):
            return f"returned error on valid input: {out[:80]}", args
    return None, None


def _repair(name, src, err, args, gen_dep):
    prompt = f"""A generated Python tool function raised an error when executed. Fix it so it runs without raising.

ERROR: {err}
CALLED WITH: {json.dumps(args)}

IMPORTANT CONTRACT: {CONTRACT}
Also be ROBUST to how the agent passes arguments: numeric/list arguments may arrive as native Python
lists/numbers (e.g. [100.0, 50.0]) OR as JSON strings. Accept both; do NOT reject valid inputs with an error.

Return ONLY the corrected function source starting with `def {name}(`. Keep the same name, signature, and the
JSON tool-schema docstring. No markdown fences, no commentary.

FUNCTION:
{src}
"""
    resp = gen_completion(model=gen_dep, messages=[{"role": "user", "content": prompt}])
    text = model_dump(resp.choices[0].message)["content"] or ""
    text = text.replace("```python", "").replace("```", "")
    i = text.find(f"def {name}")
    if i < 0:
        i = text.find("def ")
    return text[i:].strip() if i >= 0 else None


def verify_and_repair(name, src, base_server, pool, max_retries=2, verbose=True):
    """Execute -> on raise, repair via generator -> retry up to max_retries. Returns (src, ok, attempts)."""
    gen_dep = os.environ.get("LIBGEN_GEN_DEPLOYMENT", "o4-mini_2025-04-16")
    test_args = make_test_args(src, pool)
    if test_args is None:
        if verbose:
            print(f"  [verify] {name}: no test inputs available -> accepting unverified")
        return src, True, 0
    for attempt in range(max_retries + 1):
        err, args = execute_tool(base_server, src, name, test_args)
        if err is None:
            if verbose:
                print(f"  [verify] {name}: runs clean" + (f" after {attempt} repair(s)" if attempt else ""))
            return src, True, attempt
        if verbose:
            print(f"  [verify] {name}: attempt {attempt} errored -> {err[:80]}")
        if attempt == max_retries:
            break
        fixed = _repair(name, src, err, args, gen_dep)
        if not fixed:
            break
        src = fixed
        test_args = make_test_args(src, pool) or test_args
    return src, False, max_retries
