# How to Run the Tau-Bench Tool Learning Experiment

Everything below assumes you're working from the **`libgen-verify-repair`** branch:

```bash
cd C:\Users\sbarke\Downloads\tau-bench
git checkout libgen-verify-repair
```

---

## 1. Prerequisites

### 1.1 Azure CLI Login (TRAPI auth)

All inference goes through Microsoft's TRAPI endpoint, authenticated via `AzureCliCredential`.
Make sure you're logged in:

```bash
az login
```

Tokens expire after ~1 hour, but `trapi_infer.py` auto-retries on `ClientAuthenticationError`
so long runs self-heal. If you've been logged out entirely, re-run `az login`.

### 1.2 Install Python Dependencies

```bash
# Install the tau_bench package + its dependencies
pip install -e .

# Additional dependencies used by the scoring and generation scripts
pip install fastmcp mcp
```

The `setup.py` pulls in: `openai`, `mistralai`, `anthropic`, `google-generativeai`, `tenacity`,
`termcolor`, `numpy`, `litellm`.

### 1.3 Verify TRAPI Access

Quick smoke test to confirm your credentials and endpoint work:

```bash
python -c "
from tau_bench.trapi_infer import completion
r = completion(model='gpt-4o-mini', messages=[{'role':'user','content':'Say hello'}])
print(r.choices[0].message.content)
"
```

If this prints a response, you're good. If it errors on auth, re-run `az login`.

---

## 2. Environment Variables

Each script sets its own TRAPI_* vars internally, so you usually don't need to export anything.
For reference, the key variables are:

| Variable | Description | Example |
|----------|-------------|---------|
| `TRAPI_API_VERSION` | Azure OpenAI API version | `2025-03-01-preview` |
| `TRAPI_INSTANCE` | TRAPI cluster/instance | `redmond/interactive/openai` |
| `TRAPI_MODEL_NAME` | Agent model name | `gpt-4o-mini`, `gpt-4o`, `gpt-5` |
| `TRAPI_MODEL_VERSION` | Model version string | `2024-07-18` |
| `TRAPI_DEPLOYMENT_NAME` | Full deployment name | `gpt-4o-mini_2024-07-18` |
| `LIBGEN_GEN_DEPLOYMENT` | Generator model (for tool synthesis) | `o4-mini_2025-04-16` |

Available deployments: https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information

---

## 3. Phase 1 Pipeline (Step by Step)

This is the main scoring-centric experiment. Run scripts **in order** — each step's output
feeds into the next.

### Step 1: GPT-4o Base Run (Headroom Gate G1)

Measures gpt-4o's base pass rate on 50 retail train tasks at temp=0.
Takes ~20 min with `max_concurrency=8`.

```bash
python experiments/phase1/base_gpt4o.py
```

**Outputs:**
- `experiments/phase1/gpt4o_base.json` — full trajectories (checkpoint)
- `experiments/phase1/gpt4o_base_report.json` — pass rate + failure list

**Gate G1:** KEEP gpt-4o as primary if pass rate is ~30–42/50 (has headroom).
If saturated (>42), consider harder tasks or fall back to mini.

### Step 2: Harvest GPT-4o Failures

Runs tasks 50–249 to find additional gpt-4o failures, building the combined "hard set".

```bash
python experiments/phase1/harvest_gpt4o.py
```

**Outputs:**
- `experiments/phase1/gpt4o_harvest_report.json` — harvest failures + combined hard set

**Note:** This step requires Step 1's `gpt4o_base_report.json` to exist (combines failure lists).

### Step 3: GPT-4o-mini Base Run

Measures the weak agent's baseline on 50 tasks. This is the deployment target
(tools are mined from gpt-4o trajectories but deployed to gpt-4o-mini).

```bash
python experiments/phase1/base_gpt4omini.py
```

**Outputs:**
- `experiments/phase1/gpt4omini_base.json` — full trajectories
- `experiments/phase1/gpt4omini_base_report.json` — pass rate + failure list

**Expected:** ~39/50 pass rate for gpt-4o-mini base.

### Step 4: Mine Composite Tools from GPT-4o Trajectories

Uses o4-mini (the tool generator) to analyze gpt-4o's solved trajectories and propose
up to 8 composite tools that collapse repeated multi-step patterns.

```bash
python experiments/phase1/mine_from_gpt4o_traj.py
```

**Requires:** Step 1's `gpt4o_base.json` (reads solved trajectories from it).

**Outputs:**
- `experiments/phase1/mined_from_gpt4o/mined.json` — list of mined tool names + metadata
- `experiments/phase1/mined_from_gpt4o/augmented_mined.py` — augmented MCP server (base + mined tools)

### Step 5: (Optional) Deploy Unverified Mined Tools

Runs gpt-4o-mini with the raw mined tools (before repair). Useful for comparison.

```bash
python experiments/phase1/deploy_mined.py
```

**Outputs:**
- `experiments/phase1/gpt4omini_mined_deploy.json`

**Expected:** Likely worse than base (~31/50) because mined tools haven't been verified yet.

### Step 6: Verify & Repair Mined Tools

Runs each mined tool through the verify-repair loop (`scoring/verify_repair.py`):
- Executes each tool with real argument values from gpt-4o-mini trajectories
- If a tool crashes or returns an error, feeds the error back to the generator
- Up to 5 repair attempts per tool
- Drops tools that can't be fixed

```bash
python experiments/phase1/repair_mined.py
```

**Requires:**
- Step 4's `mined.json` and `augmented_mined.py`
- Step 3's `gpt4omini_base.json` (for argument pool)

**Outputs:**
- `experiments/phase1/mined_from_gpt4o/repaired.json` — kept vs dropped tools
- `experiments/phase1/mined_from_gpt4o/augmented_mined_repaired.py` — repaired MCP server

### Step 7: Deploy Verified Library to GPT-4o-mini

**This is the headline run.** Runs gpt-4o-mini with the verified+repaired tool library.

```bash
python experiments/phase1/deploy_repaired.py
```

**Outputs:**
- `experiments/phase1/gpt4omini_repaired_v3.json`

**Expected result:** base 39/50 → **verified library 46/50 (+7)**

---

## 4. Deterministic Offline Scoring

Score any run file deterministically (no live agent, no stochastic user):

```bash
# Score the augmented run
python scoring/score_tool.py experiments/phase1/gpt4omini_repaired_v3.json "repaired_v3"

# Score the base run for comparison
python scoring/score_tool.py experiments/phase1/gpt4omini_base.json "base"
```

**Outputs:** `experiments/scoring/offline/<label>.json` — per-task replay reward + agreement
with recorded reward.

### Replay Harness (programmatic)

```python
from scoring import replay as R

# Load tool functions from an MCP server (strips @mcp.tool() decorators)
funcs = R.load_server_funcs("mcp/retail_server.py")

# Seed fresh data.json, replay a trajectory, compute reward
R.seed_fresh()
actions = R.actions_from_record(record)   # [(tool_name, {kwargs}), ...]
final_data = R.replay(actions, funcs)
gold_hash = R.gold_hash(R.actions_from_gold(record["info"]), funcs)
reward = R.reward_of(final_data, gold_hash)
```

---

## 5. Full Iterative LibGen Pipeline (Config-Driven)

For the full generation→validation→test pipeline with configurable iterations:

```bash
python libgen_experiment.py --config configs/libgen/retail.json
```

This orchestrates:
1. Load baseline trajectories from `results/`
2. Split tasks into train (0–49) / validation (50–79) / test (80–99)
3. Generate candidate tools from training trajectories (3 iterations × 5 tasks each)
4. Validate on held-out tasks
5. Test surviving tools on the test split

**Config options:** See `configs/libgen/retail.json` for all knobs (iterations, chunk size,
concurrency, model settings, etc.).

Other config variants:
- `configs/libgen/airline.json` — airline domain
- `configs/libgen/retail_fast.json` — quick smoke run
- `configs/libgen/retail_smoke.json` — minimal smoke test

---

## 6. Single Benchmark Runs

Run the benchmark directly (without the libgen pipeline):

```bash
# Default: gpt-5 agent, retail, test split
python run.py --env retail --task-split test --max-concurrency 8

# Specific model
python run.py --env retail --model gpt-4o --task-split train

# With augmented MCP server (custom tools)
python run.py --env retail --mcp-server mcp/augmented_retail_server.py

# Specific task IDs only
python run.py --env retail --task-ids 0 1 2 3 4 --max-concurrency 4

# Airline domain
python run.py --env airline --task-split test
```

### Key `run.py` flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `retail` | `retail` or `airline` |
| `--task-split` | `test` | `train`, `dev`, or `test` |
| `--task-ids` | all | Specific task IDs to run |
| `--start-index` / `--end-index` | 0 / -1 | Range of tasks |
| `--mcp-server` | `mcp/retail_server.py` | Path to MCP tool server |
| `--num-trials` | 1 | Trials per task |
| `--max-concurrency` | 1 | Parallel tasks |
| `--temperature` | 0.1 | Sampling temperature |
| `--agent-strategy` | `tool-calling` | Agent type |
| `--log-dir` | `results` | Output directory |

---

## 7. Important Notes

### data.json

`data.json` in the repo root is a **runtime state blob** (~2 MB). It gets mutated during every
benchmark run. **Do not commit it.** If it looks corrupted, restore it:

```bash
git checkout -- data.json
```

### Checkpointing

All run scripts use `--ckpt-path` / `ckpt_path`. If a run crashes mid-way, re-running the
same script resumes from the checkpoint. To start fresh, delete the checkpoint JSON file first.

### Token Expiry

Azure CLI tokens expire after ~1 hour. `trapi_infer.py` retries `ClientAuthenticationError`
automatically (re-mints tokens via `AzureCliCredential`). For runs longer than ~1 hour,
this should self-heal. If it doesn't, re-run `az login` in another terminal.

### Two Inference Paths

- **Agent model** (`completion()`): controlled by `TRAPI_*` env vars (set by each script)
- **Generator model** (`gen_completion()`): controlled by `LIBGEN_GEN_*` env vars (defaults
  to gpt-5). This is the model that *designs* composite tools — it's separate from the agent.

---

## 8. File Map

| Path | Purpose |
|------|---------|
| `run.py` | Main benchmark entrypoint |
| `libgen_experiment.py` | Config-driven iterative pipeline |
| `lib_gen.py` | LLM-backed tool suggestion / generation / repair |
| `libgen_utils.py` | Metrics, trajectory parsing, MCP tool extraction |
| `tau_bench/run.py` | Benchmark executor (parallel task runner) |
| `tau_bench/trapi_infer.py` | TRAPI inference (agent + generator models) |
| `tau_bench/envs/base.py` | `calculate_reward` (deterministic DB-hash comparison) |
| `scoring/replay.py` | Deterministic offline replay harness |
| `scoring/verify_repair.py` | Execute-and-repair loop for tool verification |
| `scoring/score_tool.py` | CLI tool scoring harness |
| `mcp/retail_server.py` | Base retail tool server (23 tools) |
| `mcp/augmented_retail_server.py` | Augmented server with generated tools |
| `experiments/phase1/` | Phase 1 scripts and results |
| `experiments/libgen/` | Iterative pipeline runs and artifacts |
| `configs/libgen/` | Experiment config files |
| `GUIDE.md` | Detailed scoring-centric plan and research context |
