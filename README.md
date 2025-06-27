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

In the following commands, the agent-name can be one of the following:
Original tau bench baseline: tool-calling
Tool calling with preconditions in the form of advice: tool-calling-with-preconditions
Tool calling with preconditions in the form of advice and python code generation: tool-calling-with-preconditions-and-python
Symbolic multi-agent system: orchestrator
With pre and posty conditions: assertions-agent


To run specific tasks, use the `--task-ids` flag. For example:

```bash
python run.py --agent-strategy <agent-name> --env retail --model none --model-provider openai --user-model none --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 1
```

To run a range of tasks, use the `--start-index` and `--end-index` flags. For example:
 
 ```bash
python run.py --agent-strategy <agent-name> --env retail --model none --model-provider openai --user-model none --user-model-provider openai --user-strategy llm --max-concurrency 10 --start-index 10 --end-index 100
```

