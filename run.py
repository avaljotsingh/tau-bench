# Copyright Sierra
import os
import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-5")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-11-20")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-5_2025-08-07")

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--new-func", type=str, default=None
    )
    parser.add_argument(
        "--mcp-server", type=str, default="mcp/retail_server.py"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot", "one-shot", "assertions-agent", "orchestrator", "tool-calling-with-preconditions", "tool-calling-with-preconditions-and-python", "tool-calling-with-subtasks-check", "tool-calling-with-subtasks-feedback", "tool-calling-with-dynamic-subtasks", 'tool-calling-with-reference', 'tool-calling-with-dynamic-subtasks-with-feedback'],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    args = parser.parse_args()
    # Fallbacks to avoid validation errors when user_model_provider is omitted
    if args.user_model_provider is None:
        args.user_model_provider = args.model_provider or "openai"
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        mcp_server=args.mcp_server,
        ckpt_path=args.ckpt_path,
        new_func=args.new_func,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
