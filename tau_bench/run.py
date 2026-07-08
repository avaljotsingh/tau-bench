# Copyright Sierra

import os
import json
import time
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
from azure.ai.inference.models._models import FunctionCall

def make_serializable(obj):
    if isinstance(obj, FunctionCall):
        return obj.as_dict()

    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]

    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}

    else:
        return obj


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot", "one-shot", "assertions-agent", "orchestrator", "tool-calling-with-preconditions", "tool-calling-with-preconditions-and-python", "tool-calling-with-subtasks-check", "tool-calling-with-subtasks-feedback", "tool-calling-with-dynamic-subtasks", "tool-calling-with-reference", 'tool-calling-with-dynamic-subtasks-with-feedback'], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    if config.ckpt_path == "":  
        # sanitize model names for filesystem paths (avoid '/')
        agent_model_name = config.model.split('/')[-1] if config.model else "model"
        user_model_name = config.user_model.split('/')[-1] if config.user_model else "user_model"
        ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{agent_model_name}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{user_model_name}-{config.user_strategy}_{time_str}.json"
    else:
        ckpt_path = config.ckpt_path
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        mcp_server=config.mcp_server
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            # Build the env inside the guarded block: a transient failure here
            # (a TRAPI/network blip, or fastmcp's diskcache-store import racing when
            # many MCP-server subprocesses spawn at once) must fail only THIS task,
            # not crash the run. These races clear on retry, so retry a few times.
            isolated_env = None
            for _attempt in range(5):
                try:
                    isolated_env = get_env(
                        config.env,
                        user_strategy=config.user_strategy,
                        user_model=config.user_model,
                        task_split=config.task_split,
                        user_provider=config.user_model_provider,
                        task_index=idx,
                        mcp_server=config.mcp_server,
                    )
                    break
                except Exception as e:
                    time.sleep(1.5 * (_attempt + 1))
                    last_setup_err = e
            if isolated_env is None:
                import traceback as _tb
                print("❌", f"task_id={idx}", "env setup failed after retries:", str(last_setup_err))
                return EnvRunResult(
                    task_id=idx, reward=0.0,
                    info={"error": str(last_setup_err), "traceback": "".join(_tb.format_exception(type(last_setup_err), last_setup_err, last_setup_err.__traceback__))},
                    traj=[], trial=i, records={},
                )

            print(f"Running task {idx}")
            # res = agent.solve(
            #     env=isolated_env,
            #     task_index=idx,
            # )
            # result = EnvRunResult(
            #     task_id=idx,
            #     reward=res.reward,
            #     info=res.info,
            #     traj=res.messages,
            #     trial=i,
            #     records=res.records,
            # )
            try:
                # print(agent)
                # print(config.new_func)
                if config.new_func is not None:
                    res = agent.solve(
                        env=isolated_env,
                        new_func_name=config.new_func,
                        task_index=idx,
                    )
                else:
                    res = agent.solve(
                        env=isolated_env,
                        task_index=idx,
                    )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                    # Persist the COMPLETE action list (incl. final writes) so trajectories are
                    # replay-faithful offline; res.messages/traj is a display log that drops writes.
                    records={**(res.records or {}), "env_actions": [
                        {"name": a.name, "kwargs": a.kwargs}
                        for a in getattr(isolated_env, "actions", [])
                    ]},
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                    records={},
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            # print(result.records)
            with lock:
                data = []
                if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
                    with open(ckpt_path, "r") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to decode JSON from {ckpt_path}. Starting fresh.")
                            data = []
                with open(ckpt_path, "w") as f:
                    temp = make_serializable(result.model_dump())
                    json.dump(data + [temp], f, indent=2)
                return result

        # Run tasks concurrently. _run is thread-safe: each task gets an isolated
        # env, the shared agent is only read, and checkpoint writes are guarded by
        # `lock`. max_concurrency caps the number of in-flight tasks.
        # Collect per-future so that ANY unexpected exception escaping a single
        # task is recorded as a failed task instead of aborting the whole run.
        def _safe_run(idx: int) -> EnvRunResult:
            try:
                return _run(idx)
            except Exception as e:
                print("❌", f"task_id={idx}", "uncaught task error:", str(e))
                return EnvRunResult(
                    task_id=idx, reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[], trial=i, records={},
                )

        if config.max_concurrency and config.max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
                results.extend(executor.map(_safe_run, idxs))
        else:
            for idx in idxs:
                results.append(_safe_run(idx))

    display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([make_serializable(result.model_dump()) for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-preconditions":
        # tool calling with preconditions
        from tau_bench.agents.tool_calling_with_preconditions import ToolCallingAgentWithPreconditions

        return ToolCallingAgentWithPreconditions(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-reference":
        # tool calling with reference
        from tau_bench.agents.tool_calling_with_reference import ToolCallingAgentWithReference

        return ToolCallingAgentWithReference(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-subtasks-check":
        # tool calling with preconditions
        from tau_bench.agents.tool_calling_with_subtasks_check import ToolCallingWithSubtasksCheckAgent

        return ToolCallingWithSubtasksCheckAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-subtasks-feedback":
        # tool calling with preconditions
        from tau_bench.agents.tool_calling_with_subtasks_feedback import ToolCallingWithSubtasksFeedbackAgent

        return ToolCallingWithSubtasksFeedbackAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-dynamic-subtasks":
        # tool calling with preconditions
        from tau_bench.agents.tool_calling_with_dynamic_subtasks import ToolCallingWithDynamicSubtasks

        return ToolCallingWithDynamicSubtasks(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-dynamic-subtasks-with-feedback":
        # tool calling with preconditions
        from tau_bench.agents.tool_calling_with_dynamic_subtasks_with_feedback import ToolCallingWithDynamicSubtasksWithFeedback

        return ToolCallingWithDynamicSubtasksWithFeedback(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "tool-calling-with-preconditions-and-python":
        # tool calling with preconditions
        from tau_bench.agents.tool_calling_with_preconditions_and_python import ToolCallingAgentWithPreconditionsAndPython

        return ToolCallingAgentWithPreconditionsAndPython(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "one-shot":
        # native tool calling
        from tau_bench.agents.one_shot_agent import OneShotAgent

        return OneShotAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "assertions-agent":
        # native tool calling
        from tau_bench.agents.assertions_agent import AssertionsAgent

        return AssertionsAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "orchestrator":
        # native tool calling
        from tau_bench.agents.orchestrator import Orchestrator

        return Orchestrator(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
