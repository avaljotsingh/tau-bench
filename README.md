# AgentVerify

## Setup

1. Clone this repository:

```bash
git clone https://github.com/avaljotsingh/tau-bench.git && cd ./tau-bench
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```


## Run

To run specific tasks, use the `--task-ids` flag. For example:

```bash
python run.py --agent-strategy assertions-agent --env retail --model none --model-provider openai --user-model none --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 1
```
