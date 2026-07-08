"""Deterministic offline replay harness for scoring generated composite tools.

Reward in tau-bench is `data_hash == gt_data_hash` (tau_bench/envs/base.py) — a pure function of the final
DB state. So we can score a tool by replaying a (rewritten) action sequence with NO live agent and NO
simulated user, fully deterministically.

Mechanism (all in the MCP tool layer, which persists state via ./data.json):
  1. seed data.json with fresh retail data (task-independent initial state)
  2. replay a list of (tool_name, kwargs) actions by calling the plain tool functions
  3. hash the resulting data with the env's own consistent_hash and compare to the gold-action replay

Tool functions are loaded from a server module (e.g. experiments/weak_experiment/augmented.py) by stripping
the @mcp.tool() decorators so we get plain callables (no FastMCP runtime).
"""
import os, re, json, types
from typing import Any, Dict, List, Tuple, Callable

from tau_bench.envs.base import to_hashable, consistent_hash
from tau_bench.envs.retail.data import load_data

DATA_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.json")

RESPOND = "respond"  # RESPOND_ACTION_NAME; never mutates data


# --------------------------------------------------------------------------- loading tool functions
def load_server_funcs(server_path: str) -> Dict[str, Callable]:
    """Exec a server module's source with @mcp.tool() stripped, returning {name: plain_function}."""
    src = open(server_path, encoding="latin-1").read()
    # Drop the FastMCP wiring and decorators so bodies become plain module-level functions.
    src = re.sub(r"^\s*from mcp\.server\.fastmcp import FastMCP.*$", "", src, flags=re.M)
    src = re.sub(r"^\s*mcp\s*=\s*FastMCP\(.*\).*$", "mcp = None", src, flags=re.M)
    src = re.sub(r"^\s*@mcp\.tool\(\).*$", "", src, flags=re.M)
    ns: Dict[str, Any] = {}
    exec(compile(src, server_path, "exec"), ns)
    return {k: v for k, v in ns.items() if isinstance(v, types.FunctionType) and not k.startswith("_")}


# --------------------------------------------------------------------------- data.json state
def seed_fresh() -> Dict[str, Any]:
    data = load_data()
    with open(DATA_JSON, "w") as f:
        json.dump(data, f)
    return data


def read_data() -> Dict[str, Any]:
    with open(DATA_JSON, "r") as f:
        return json.load(f)


def data_hash(data: Dict[str, Any]) -> str:
    return consistent_hash(to_hashable(data))


# --------------------------------------------------------------------------- replay
def replay(actions: List[Tuple[str, dict]], funcs: Dict[str, Callable], strict: bool = False) -> Dict[str, Any]:
    """Seed fresh data.json, then apply each write/read action by calling its tool function.

    Read actions don't mutate data.json but are harmless. Unknown tools / RESPOND are skipped.
    Returns the final data dict. If strict, re-raise tool errors instead of swallowing them.
    """
    seed_fresh()
    for name, kwargs in actions:
        if name == RESPOND or name not in funcs:
            continue
        try:
            funcs[name](**kwargs)
        except Exception as e:
            if strict:
                raise
            # tool errors are part of normal trajectories (e.g. bad id) — keep going
    return read_data()


def gold_hash(task_actions: List[Tuple[str, dict]], funcs: Dict[str, Callable]) -> str:
    """Ground-truth data hash = replay the task's gold actions on fresh data."""
    return data_hash(replay([a for a in task_actions if a[0] != RESPOND], funcs))


def reward_of(final_data: Dict[str, Any], gt_hash: str) -> float:
    """DB-state (r_actions) component only. Prefer reward_full when the task has required outputs."""
    return 1.0 if data_hash(final_data) == gt_hash else 0.0


def outputs_satisfied(task_outputs: List[str], actions: List[Tuple[str, dict]]) -> bool:
    """r_outputs: every required output string must appear in some RESPOND content (lc, commas stripped)."""
    for output in task_outputs:
        found = any(
            name == RESPOND and output.lower() in str(kw.get("content", "")).lower().replace(",", "")
            for name, kw in actions
        )
        if not found:
            return False
    return True


def reward_full(final_data: Dict[str, Any], gt_hash: str,
                task_outputs: List[str], actions: List[Tuple[str, dict]]) -> float:
    """Full tau-bench reward = r_actions (DB hash) AND r_outputs (required strings said to the user)."""
    if data_hash(final_data) != gt_hash:
        return 0.0
    if task_outputs and not outputs_satisfied(task_outputs, actions):
        return 0.0
    return 1.0


def task_outputs_of(task_info: dict) -> List[str]:
    return (task_info.get("task") or {}).get("outputs", []) or []


# --------------------------------------------------------------------------- trajectory parsing
def actions_from_traj(traj: List[dict]) -> List[Tuple[str, dict]]:
    """Extract the agent's (tool_name, kwargs) sequence from a stored `traj` message list."""
    out: List[Tuple[str, dict]] = []
    for m in traj:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            raw = fn.get("arguments", "{}")
            try:
                kwargs = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                kwargs = {}
            out.append((name, kwargs))
    return out


def actions_from_record(rec: dict) -> List[Tuple[str, dict]]:
    """Prefer the complete env_actions log (replay-faithful); fall back to the display traj."""
    ea = (rec.get("records") or {}).get("env_actions")
    if ea:
        return [(a["name"], a.get("kwargs", {})) for a in ea]
    return actions_from_traj(rec.get("traj", []))


def actions_from_gold(task_info: dict) -> List[Tuple[str, dict]]:
    """Extract gold (tool_name, kwargs) from a record's info.task.actions."""
    acts = (task_info.get("task") or {}).get("actions", [])
    return [(a["name"], a.get("kwargs", {})) for a in acts]
