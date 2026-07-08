"""Deploy the FULL mined library (no pre-elimination) on gpt-4o-mini and measure what actually happens.

Exposes all 7 tools mined from gpt-4o's trajectories to the gpt-4o-mini agent over the 50 retail tasks.
Saves complete env_actions so we can measure, per tool, REAL adoption (did the agent call it?) and
deterministic correctness of whatever it used. Compare reward + adoption vs the base run (39/50).
"""
import os, sys, json
os.environ["TRAPI_API_VERSION"] = "2025-03-01-preview"
os.environ["TRAPI_INSTANCE"] = "redmond/interactive/openai"
os.environ["TRAPI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["TRAPI_MODEL_VERSION"] = "2024-07-18"
os.environ["TRAPI_DEPLOYMENT_NAME"] = "gpt-4o-mini_2024-07-18"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run

N = 50
OUT = "experiments/phase1"
CKPT = os.path.join(OUT, "gpt4omini_mined_deploy.json")
SERVER = "experiments/phase1/mined_from_gpt4o/augmented_mined.py"

cfg = RunConfig(
    model_provider="openai", user_model_provider="openai", model="none", user_model="none",
    num_trials=1, env="retail", agent_strategy="tool-calling", temperature=0.0, task_split="train",
    start_index=0, end_index=-1, task_ids=list(range(N)), log_dir=OUT, max_concurrency=8, seed=10,
    shuffle=0, user_strategy="llm", few_shot_displays_path=None, mcp_server=SERVER,
    ckpt_path=CKPT, new_func=None)

res = tau_run(cfg)
passed = sum(1 for r in res if r.reward >= 1 - 1e-6)
print(f"\n=== gpt-4o-mini + 7 mined tools: {passed}/{N} (base was 39/50) -> {CKPT} ===")
