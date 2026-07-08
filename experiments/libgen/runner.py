import os
import json
from typing import Any, Dict, List, Tuple

from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-5_2025-08-07"
INSTANCE = "gcr/shared"
ENDPOINT = f"https://trapi.research.microsoft.com/{INSTANCE}"

import lib_gen as lib_gen
from libgen_utils import (
    get_tools,
    Library,
    create_file,
    Metrics,
    extract_docstring_from_function_string,
    is_docstring_json,
)

from .utils import (
    ExperimentLogger,
    ExperimentPaths,
    copy_file,
    ensure_dir,
    extract_minimal_tasks_from_results,
    find_latest_results,
    load_json,
    normalize_ids,
    save_json,
    validate_config,
)


class LibGenExperimentRunner:
    def __init__(self, config: Dict[str, Any]) -> None:
        validate_config(config)
        self.config = config
        io_cfg = config["io"]
        self.paths = ExperimentPaths(
            root=io_cfg["experiments_root"],
            name=config["experiment_name"],
            mcp_subdir=io_cfg["mcp_subdir"],
            log_dir_name=io_cfg["log_dir"],
        )
        ensure_dir(self.paths.root)
        ensure_dir(self.paths.logs_dir)
        ensure_dir(self.paths.metrics_dir)
        ensure_dir(self.paths.artifacts_dir)
        ensure_dir(self.paths.functions_dir)
        ensure_dir(self.paths.mcp_out_dir)
        ensure_dir(self.paths.phases_dir)
        ensure_dir(self.paths.ckpt_dir)
        ensure_dir(self.paths.input_dir)
        self.logger = ExperimentLogger(self.paths.experiment_log_path())
        # Snapshot manifest
        manifest = {
            "config": config,
        }
        save_json(self.paths.manifest_path(), manifest)

    def _env_models(self) -> Tuple[str, str]:
        agent_model = self.config["agent"].get("model") or os.environ.get("LIBGEN_AGENT_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("LIBGEN_MODEL") or "none"
        user_model = self.config["agent"].get("user_model") or os.environ.get("LIBGEN_USER_MODEL") or agent_model
        return agent_model, user_model

    def _build_run_config(
        self,
        task_ids: List[int],
        task_split: str,
        mcp_server: str,
        ckpt_path: str,
        new_func: str = None,
    ) -> RunConfig:
        agent = self.config["agent"]
        agent_model, user_model = self._env_models()
        rc = RunConfig(
            model_provider=agent["model_provider"],
            user_model_provider=agent["user_model_provider"],
            model=agent_model,
            user_model=user_model,
            num_trials=agent.get("num_trials", 1),
            env=self.config["env"],
            agent_strategy=agent["agent_strategy"],
            temperature=agent.get("temperature", 0.0),
            task_split=task_split,
            start_index=0,
            end_index=-1,
            task_ids=task_ids,
            log_dir=self.paths.ckpt_dir,
            max_concurrency=agent.get("max_concurrency", 1),
            seed=self.config["runner"].get("seed", 42),
            shuffle=0,
            user_strategy=agent.get("user_strategy", "llm"),
            few_shot_displays_path=agent.get("few_shot_displays_path"),
            mcp_server=mcp_server,
            ckpt_path=ckpt_path,
            new_func=new_func,
        )
        return rc

    def _resolve_input_tasks(self) -> List[Dict[str, Any]]:
        input_cfg = self.config["input_tasks"]
        env = self.config["env"]
        explicit = input_cfg.get("explicit_path")
        if explicit:
            self.logger.log(f"Using explicit input tasks file: {explicit}")
            tasks = extract_minimal_tasks_from_results(explicit)
            save_json(self.paths.input_tasks_path(), tasks)
            return tasks
        # Mode selection
        if input_cfg["mode"] == "latest_for_env":
            logs_dir = input_cfg.get("logs_dir", "results")
            filename_glob = input_cfg.get("filename_glob", "*.json")
            validate_env = bool(input_cfg.get("validate_env_in_file", True))
            latest = find_latest_results(logs_dir, filename_glob, env, validate_env)
            if latest is None:
                raise FileNotFoundError(f"No results found in {logs_dir} for mode latest_for_env")
            self.logger.log(f"Detected latest results file: {latest}")
            tasks = extract_minimal_tasks_from_results(latest)
            save_json(self.paths.input_tasks_path(), tasks)
            return tasks
        elif input_cfg["mode"] == "base_library":
            # Warmup to collect tasks
            return self._warmup_collect_tasks()
        else:
            raise ValueError("Unknown input_tasks.mode")

    def _warmup_collect_tasks(self) -> List[Dict[str, Any]]:
        library_cfg = self.config["library"]
        if not library_cfg.get("base_library_path"):
            raise ValueError("library.base_library_path is required for base_library mode")
        base_mcp = library_cfg["base_library_path"]
        copy_file(base_mcp, self.paths.base_library_copy_path())
        ensure_dir(self.paths.warmup_dir())
        warmup_ckpt = os.path.join(self.paths.warmup_dir(), "ckpt.json")
        train_ids = normalize_ids(self.config["splits"]["train_ids"])
        self.logger.log(f"Starting warmup run to collect tasks. Train IDs: {train_ids}")
        rc = self._build_run_config(
            task_ids=train_ids,
            task_split="train",
            mcp_server=base_mcp,
            ckpt_path=warmup_ckpt,
        )
        tau_run(rc)
        tasks = extract_minimal_tasks_from_results(warmup_ckpt)
        save_json(self.paths.input_tasks_path(), tasks)
        self.logger.log(f"Warmup completed. Extracted {len(tasks)} tasks to {self.paths.input_tasks_path()}")
        return tasks

    def _library_from_mcp(self, mcp_server: str) -> List[str]:
        tools = get_tools(mcp_server)
        return Library(tools).get_funcs()

    def _generate_func(
        self,
        tasks: List[Dict[str, Any]],
        task_ids: List[int],
        mcp_server_before_generation: str,
        new_mcp_server: str,
        output_folder: str,
    ) -> Tuple[bool, str, str]:
        old_library = self._library_from_mcp(mcp_server_before_generation)
        # propose
        def_flag = False
        doc_trial = 0
        new_func_name = None
        new_func_def = None
        while not def_flag and doc_trial < 2:
            doc_trial += 1
            new_func_name, new_func_def = lib_gen.get_new_func(tasks, old_library)
            doc = extract_docstring_from_function_string(new_func_def)
            if is_docstring_json(doc):
                def_flag = True
        if not def_flag:
            self.logger.log("Failed to generate a valid function definition.")
            return False, None, None
        self.logger.log(f"New function proposed: {new_func_name}")
        new_func_def_history = [{"role": "user", "content": "Original function: \n" + new_func_def}]
        # refine via trajectories
        trial = 0
        found_incorrect_trajectory = True
        max_refinement_tries = int(self.config["runner"].get("max_refinement_tries", 2))
        while found_incorrect_trajectory and trial < max_refinement_tries:
            trial += 1
            new_library = old_library + [new_func_def]
            # create temporary MCP with new function
            open(new_mcp_server, "w").close()
            create_file(mcp_server_before_generation, new_mcp_server, new_func_def)
            found_incorrect_trajectory = False
            # Run all trajectory checks for this chunk concurrently, then pick the
            # FIRST failing trajectory (in task order) to drive correction -- this
            # preserves the original sequential "correct on first failure" semantics.
            self.logger.log(f"Trajectory check on tasks {list(task_ids)}")
            output_file = os.path.join(output_folder, "train_results.json")
            rc = self._build_run_config(
                task_ids=list(task_ids),
                task_split="train",
                mcp_server=new_mcp_server,
                ckpt_path=output_file,
                new_func=new_func_name,
            )
            tau_run(rc)
            trajectories = load_json(output_file)
            by_id = {t.get("task_id"): t for t in trajectories}
            new_trajectory = None
            failing_task_id = None
            for task_id in task_ids:
                trajectory = by_id.get(task_id)
                if trajectory is None:
                    continue
                res = Metrics(trajectory, [new_func_name])
                if res.error_in_func[new_func_name] is True:
                    found_incorrect_trajectory = True
                    new_trajectory = trajectory
                    failing_task_id = task_id
                    break
            if found_incorrect_trajectory:
                self.logger.log(f"Wrong trajectory detected on task {failing_task_id}")
                improved_func_def, explanation = lib_gen.correct_func_from_traj(
                    old_library, new_func_def, new_trajectory, new_func_def_history
                )
                new_func_def = improved_func_def
                self.logger.log(f"Corrected proposed function. Explanation: {explanation}")
                new_func_def_history.append({"role": "user", "content": f"Failed Trajectory: \n{new_trajectory}"})
                new_func_def_history.append(
                    {"role": "assistant", "content": f"Corrected function: \n{new_func_def}, Explanation: \n{explanation}"}
                )
        return not found_incorrect_trajectory, new_func_name, new_func_def

    def _generation_phase(
        self,
        tasks: List[Dict[str, Any]],
        task_ids: List[int],
        mcp_server_before_generation: str,
        new_mcp_server: str,
        output_folder: str,
        num_iterations: int,
    ) -> Tuple[bool, Dict[str, str]]:
        ensure_dir(output_folder)
        overall_success = False
        new_funcs: Dict[str, str] = {}
        for i in range(num_iterations):
            self.logger.log(f"Generation iteration {i+1}")
            try:
                success, new_func_name, new_func_def = self._generate_func(
                    tasks, task_ids, mcp_server_before_generation, new_mcp_server, output_folder
                )
            except Exception as e:
                # A generation/correction LLM call failed even after retries (e.g. a
                # prolonged TRAPI outage). Skip this proposal rather than crashing the
                # whole run; subsequent iterations can still succeed.
                self.logger.log(f"Generation iteration {i+1} failed with error: {e}; skipping")
                continue
            if success:
                # keep only names in temp server for iterative proposals
                open(mcp_server_before_generation, "w").close()
                create_file(new_mcp_server, mcp_server_before_generation, new_func_name)
                new_funcs[new_func_name] = new_func_def
                overall_success = True
        # store proposed functions
        if new_funcs:
            ensure_dir(self.paths.functions_dir)
            for name, code in new_funcs.items():
                with open(os.path.join(self.paths.functions_dir, f"{name}.txt"), "w") as f:
                    f.write(code)
        return overall_success, new_funcs

    def _validation_phase(
        self,
        task_ids: List[int],
        mcp_server: str,
        new_funcs: Dict[str, str],
        output_folder: str,
    ) -> List[str]:
        ensure_dir(output_folder)
        # Run all validation tasks in a single tau_run call so they execute
        # concurrently (max_concurrency) instead of one-at-a-time.
        output_file = os.path.join(output_folder, "validation_results.json")
        rc = self._build_run_config(
            task_ids=list(task_ids),
            task_split="train",
            mcp_server=mcp_server,
            ckpt_path=output_file,
        )
        tau_run(rc)
        trajectories = load_json(output_file)
        results: List[Dict[str, Any]] = [
            Metrics(trajectory, list(new_funcs.keys())).to_dict() for trajectory in trajectories
        ]
        results_file = os.path.join(output_folder, "metric_results.json")
        save_json(results_file, results)
        # consolidate
        func_called = {func: False for func in new_funcs}
        error_in_func = {func: False for func in new_funcs}
        num_calls = {func: 0 for func in new_funcs}
        for res in results:
            for func in new_funcs:
                if res["func_called"][func]:
                    func_called[func] = True
                    num_calls[func] += res["num_calls"][func]
                    if res["error_in_func"][func]:
                        error_in_func[func] = True
        filtered = []
        for func in new_funcs:
            if func_called[func] and not error_in_func[func]:
                filtered.append(func)
        save_json(os.path.join(output_folder, "selected_funcs.json"), filtered)
        return filtered

    def _test_phase(
        self,
        task_ids: List[int],
        mcp_server: str,
        new_funcs: List[str],
        output_folder: str,
    ) -> None:
        ensure_dir(output_folder)
        # Run all test tasks in a single tau_run call so they execute concurrently.
        self.logger.log(f"Test tasks {list(task_ids)}")
        output_file = os.path.join(output_folder, "test_results_raw.json")
        rc = self._build_run_config(
            task_ids=list(task_ids),
            task_split="test",
            mcp_server=mcp_server,
            ckpt_path=output_file,
        )
        tau_run(rc)
        results: List[Dict[str, Any]] = load_json(output_file)
        save_json(os.path.join(output_folder, "test_results.json"), results)

    def run(self) -> None:
        # Resolve initial tasks set
        all_tasks = self._resolve_input_tasks()
        # Splits
        train_ids = normalize_ids(self.config["splits"]["train_ids"])
        validation_ids = normalize_ids(self.config["splits"]["validation_ids"])
        test_ids = normalize_ids(self.config["splits"]["test_ids"])
        # Runner params
        gen_iters = int(self.config["runner"].get("generation_phase_iterations", 3))
        chunk_size = int(self.config["runner"].get("generation_phase_chunk_size", 5))
        test_after = int(self.config["runner"].get("test_after_iterations", 6))
        # Initialize MCP before generation as base library (copy base or latest)
        mcp_server_before_generation = None
        library_cfg = self.config["library"]
        if library_cfg["mode"] == "base_library":
            mcp_server_before_generation = library_cfg["base_library_path"]
        else:
            # fallback: use copied base if provided, else user should supply
            mcp_server_before_generation = library_cfg.get("base_library_path") or ""
        if not mcp_server_before_generation or not os.path.exists(mcp_server_before_generation):
            raise FileNotFoundError("A valid library.base_library_path is required to bootstrap MCP server state")
        # Iterate updates
        # Default the post-validation server to the base library so the test
        # phase always has a valid MCP server even if no function is generated
        # or passes validation (otherwise mcp_after_validation is unbound).
        mcp_after_validation = mcp_server_before_generation
        iteration = 0
        while iteration < 10:
            for j in range(test_after):
                iteration += 1
                it_dir = self.paths.iteration_dir(iteration)
                ensure_dir(it_dir)
                self.logger.log(f"Iteration {iteration}")
                # Generation
                gen_dir = self.paths.generation_dir(iteration)
                ensure_dir(gen_dir)
                mcp_after_generation = os.path.join(self.paths.mcp_out_dir, f"mcp_after_generation_iter_{iteration}.py")
                open(mcp_after_generation, "w").close()
                # Select chunk of training tasks and IDs
                start = iteration * chunk_size
                end = (iteration + 1) * chunk_size
                tasks_chunk = all_tasks[start:end]
                train_ids_chunk = train_ids[start:end]
                if len(tasks_chunk) == 0 or len(train_ids_chunk) == 0:
                    self.logger.log("No more train tasks to process in generation phase.")
                    break
                self.logger.log("Running generation phase")
                success, new_funcs = self._generation_phase(
                    tasks_chunk,
                    train_ids_chunk,
                    mcp_server_before_generation,
                    mcp_after_generation,
                    gen_dir,
                    gen_iters,
                )
                if not success:
                    self.logger.log("Generation produced no valid functions; continuing to next iteration")
                    continue
                # Validation
                self.logger.log("Running validation phase")
                val_dir = self.paths.validation_dir(iteration)
                ensure_dir(val_dir)
                filtered_funcs = self._validation_phase(validation_ids, mcp_after_generation, new_funcs, val_dir)
                if len(filtered_funcs) == 0:
                    self.logger.log("No functions passed validation; continuing to next iteration")
                    continue
                # Stitch selected functions into a new MCP server
                mcp_after_validation = os.path.join(self.paths.mcp_out_dir, f"mcp_after_validation_iter_{iteration}.py")
                open(mcp_after_validation, "w").close()
                # temp server to iteratively build file
                mcp_temp = os.path.join(self.paths.mcp_out_dir, f"mcp_temp_iter_{iteration}.py")
                open(mcp_temp, "w").close()
                # seed temp with base
                create_file(mcp_server_before_generation, mcp_temp, "")
                for func in filtered_funcs:
                    create_file(mcp_temp, mcp_after_validation, new_funcs[func])
                    create_file(mcp_after_validation, mcp_temp, "")
                # carry over to next iteration
                mcp_server_before_generation = os.path.join(self.paths.mcp_out_dir, f"mcp_before_generation_iter_{iteration+1}.py")
                create_file(mcp_after_validation, mcp_server_before_generation, "")
                if iteration % test_after == 0:
                    break
            # Test phase
            self.logger.log("Running test phase")
            test_output_folder = self.paths.test_dir
            ensure_dir(test_output_folder)
            # filtered_funcs from last loop scope; handle if undefined
            if "filtered_funcs" not in locals() or len(filtered_funcs) == 0:
                filtered_funcs = []
            self._test_phase(test_ids, mcp_after_validation, filtered_funcs, test_output_folder)
            # end after first full cycle to avoid infinite loop
            break


