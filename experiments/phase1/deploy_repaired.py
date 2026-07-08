import os, sys, json
os.environ.update({"TRAPI_API_VERSION":"2025-03-01-preview","TRAPI_INSTANCE":"redmond/interactive/openai",
    "TRAPI_MODEL_NAME":"gpt-4o-mini","TRAPI_MODEL_VERSION":"2024-07-18","TRAPI_DEPLOYMENT_NAME":"gpt-4o-mini_2024-07-18"})
sys.path.insert(0, os.getcwd())
from tau_bench.types import RunConfig
from tau_bench.run import run as tau_run
CKPT="experiments/phase1/gpt4omini_repaired_v3.json"
if os.path.exists(CKPT): os.remove(CKPT)   # avoid checkpoint pollution
cfg=RunConfig(model_provider="openai",user_model_provider="openai",model="none",user_model="none",num_trials=1,
    env="retail",agent_strategy="tool-calling",temperature=0.0,task_split="train",start_index=0,end_index=-1,
    task_ids=list(range(50)),log_dir="experiments/phase1",max_concurrency=8,seed=10,shuffle=0,user_strategy="llm",
    few_shot_displays_path=None,mcp_server="experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py",
    ckpt_path=CKPT,new_func=None)
res=tau_run(cfg); p=sum(1 for r in res if r.reward>=1-1e-6)
print(f"RESULT_LINE gpt-4o-mini + verified library (7 tools): {p}/50  (base 39, broken 31, prev-repaired 44)")
