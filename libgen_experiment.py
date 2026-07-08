import argparse
import os
import sys
from experiments.libgen.runner import LibGenExperimentRunner
from experiments.libgen.utils import load_json
os.environ.setdefault("TRAPI_API_VERSION", "2025-03-01-preview")
os.environ.setdefault("TRAPI_INSTANCE", "redmond/interactive/openai")
os.environ.setdefault("TRAPI_MODEL_NAME", "gpt-5")
os.environ.setdefault("TRAPI_MODEL_VERSION", "2024-11-20")
os.environ.setdefault("TRAPI_DEPLOYMENT_NAME", "gpt-5_2025-08-07")
os.environ.setdefault("LIBGEN_AGENT_MODEL", "$TRAPI_MODEL_NAME")
os.environ.setdefault("LIBGEN_USER_MODEL", "$LIBGEN_AGENT_MODEL")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized LibGen Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    config = load_json(config_path)
    runner = LibGenExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()