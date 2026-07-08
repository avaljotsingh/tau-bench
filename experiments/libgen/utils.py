import json
import os
import shutil
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def validate_config(config: Dict[str, Any]) -> None:
    required_top = ["experiment_name", "env", "input_tasks", "library", "splits", "runner", "agent", "io"]
    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required config field: {key}")
    if config["env"] not in ["retail", "airline"]:
        raise ValueError("env must be one of: retail, airline")
    if config["input_tasks"]["mode"] not in ["latest_for_env", "base_library"]:
        raise ValueError("input_tasks.mode must be one of: latest_for_env, base_library")
    if config["library"]["mode"] not in ["base_library", "latest_for_env"]:
        # allow symmetry but typically base_library
        raise ValueError("library.mode must be one of: base_library, latest_for_env")
    # Splits must be lists of two ints [start, end) or explicit list of ints
    for split_key in ["train_ids", "validation_ids", "test_ids"]:
        if split_key not in config["splits"]:
            raise ValueError(f"Missing splits.{split_key}")
        _ = normalize_ids(config["splits"][split_key])
    # Agent minimal checks
    if "agent_strategy" not in config["agent"]:
        raise ValueError("agent.agent_strategy is required")
    if "model_provider" not in config["agent"]:
        raise ValueError("agent.model_provider is required")
    if "user_model_provider" not in config["agent"]:
        raise ValueError("agent.user_model_provider is required")
    # IO required fields
    for key in ["experiments_root", "mcp_subdir", "log_dir"]:
        if key not in config["io"]:
            raise ValueError(f"Missing io.{key}")


def normalize_ids(ids_or_range: Union[List[int], Tuple[int, int]]) -> List[int]:
    if (
        isinstance(ids_or_range, list)
        and len(ids_or_range) == 2
        and all(isinstance(x, int) for x in ids_or_range)
    ):
        start, end = ids_or_range
        return list(range(start, end))
    if isinstance(ids_or_range, list) and all(isinstance(x, int) for x in ids_or_range):
        return ids_or_range
    raise ValueError("IDs must be [start, end) or a list of integers")


# Tool-name markers used to infer the benchmark domain of a results file when
# the file does not record an explicit env (the tau-bench result `info` block
# typically does not). Detection is based on which environment's tools appear
# in the recorded trajectories.
_ENV_TOOL_MARKERS = {
    "retail": (
        "find_user_id_by_name_zip",
        "get_product_details",
        "get_order_details",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "list_all_product_types",
    ),
    "airline": (
        "book_reservation",
        "search_direct_flight",
        "search_onestop_flight",
        "update_reservation_passengers",
        "update_reservation_flights",
        "get_reservation_details",
    ),
}


def detect_env_in_results(path: str) -> Optional[str]:
    """Infer the benchmark env ('retail'/'airline') from a results file.

    Prefers an explicit env recorded in the result `info` block; otherwise
    falls back to counting environment-specific tool names in the raw file
    text and returns the dominant env (None if neither/ambiguous).
    """
    try:
        data = load_json(path)
    except Exception:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    info = data[0].get("info", {}) if isinstance(data[0], dict) else {}
    explicit = info.get("env") or info.get("environment")
    if explicit in _ENV_TOOL_MARKERS:
        return explicit
    blob = json.dumps(data)
    counts = {
        env: sum(blob.count(marker) for marker in markers)
        for env, markers in _ENV_TOOL_MARKERS.items()
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return None
    # Require a clear majority to avoid mislabeling mixed-domain files.
    others = sum(v for k, v in counts.items() if k != best)
    if counts[best] <= others:
        return None
    return best


def find_latest_results(
    logs_dir: str,
    filename_glob: str,
    env: str,
    validate_env_in_file: bool,
) -> Optional[str]:
    candidates = sorted(
        glob(os.path.join(logs_dir, filename_glob)),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        return None
    if not validate_env_in_file:
        return candidates[0]
    for path in candidates:
        if detect_env_in_results(path) == env:
            return path
    # No file matched the requested env: fail loudly rather than silently
    # feeding the wrong domain's trajectories into generation.
    return None


def extract_minimal_tasks_from_results(results_path: str) -> List[Dict[str, Any]]:
    data = load_json(results_path)
    tasks: List[Dict[str, Any]] = []
    for item in data:
        tasks.append(
            {
                "task_id": item.get("task_id"),
                "traj": item.get("traj", []),
                "reward": item.get("reward", 0.0),
                "info": item.get("info", {}),
            }
        )
    return tasks


class ExperimentPaths:
    def __init__(self, root: str, name: str, mcp_subdir: str, log_dir_name: str):
        self.root = os.path.join(root, name)
        self.input_dir = os.path.join(self.root, "input")
        self.logs_dir = os.path.join(self.root, "logs")
        self.metrics_dir = os.path.join(self.root, "metrics")
        self.test_dir = os.path.join(self.root, "test")
        self.phases_dir = os.path.join(self.root, "phases")
        self.artifacts_dir = os.path.join(self.root, "artifacts")
        self.functions_dir = os.path.join(self.artifacts_dir, "functions")
        self.mcp_out_dir = os.path.join(self.artifacts_dir, mcp_subdir)
        self.ckpt_dir = os.path.join(self.root, log_dir_name)

    def iteration_dir(self, i: int) -> str:
        return os.path.join(self.phases_dir, f"iteration_{i}")

    def generation_dir(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "generation")

    def validation_dir(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "validation")

    def warmup_dir(self) -> str:
        return os.path.join(self.root, "phases", "warmup")

    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    def input_tasks_path(self) -> str:
        return os.path.join(self.input_dir, "tasks.json")

    def base_library_copy_path(self) -> str:
        return os.path.join(self.input_dir, "library_base.py")

    def experiment_log_path(self) -> str:
        return os.path.join(self.logs_dir, "experiment.log")


class ExperimentLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(os.path.dirname(self.log_path))

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")


