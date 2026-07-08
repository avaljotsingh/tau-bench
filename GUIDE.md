# Tau-Bench Libgen Overview

This repository studies how to improve tool-calling agents by generating better tools from observed trajectories.

The main research loop is:

1. Run a tool-calling agent on benchmark tasks and collect its trajectories.
2. Look for repeated multi-step patterns, such as repeated lookups or filtering loops.
3. Propose a composite tool that would collapse those steps into a single call.
4. Generate and refine that tool from the training trajectories.
5. Validate the tool on held-out tasks.
6. Keep the tools that help on validation and evaluate them on the test split.

The repository contains the active generation pipeline, the `tau_bench` benchmark runtime, snapshots of MCP servers created during experiments, baseline trajectories, and several analysis-oriented folders and notebooks.

Concrete example:

Suppose a user asks, “I want to return the shoes from my last order.” The agent may only have low-level tools such as `get_user_details` and `get_order_details`.

The current tool-calling pattern might look like this:

1. The agent calls `get_user_details(user_id)`.
2. That response contains a list of several orders.
3. The agent inspects those orders and finds that there are multiple candidates.
4. The agent calls `get_order_details(order_id)` for each candidate order.
5. It compares the line items until it finds the order that contains the shoes.
6. It then continues with the return flow for the matching order.

This works, but it is inefficient because the agent is manually doing the same lookup logic every time.

A better composite tool would encode that repeated search directly, for example:

- `find_order_by_item(user_id, item_name)`

With that tool, the agent can do the same job in one call:

1. Call `find_order_by_item(user_id, "shoes")`.
2. Receive the matching order immediately.
3. Continue directly to the return action.

That is the general goal of the repository: inspect trajectories, notice repeated decision patterns, and turn them into reusable higher-level tools that reduce the number of steps the agent needs.

---

# Where We Are Now (status — read before planning)

> **[REVISE]** tags below mark knobs you may want to change.

The generation mechanism works — it produces plausible, adopted composite tools. **But mined tools have not
improved τ-bench retail, and the reason turned out to be the *scoring*, not the generation.** Honest ledger
(each row from a committed `experiments/*/report.json`):

| Experiment | Model | Result |
|---|---|---|
| Solved-trajectory A/B | gpt-5 | base 20/20 @ 7.95 calls → aug **18/20 @ +27% calls**. Tools adopted, hurt both axes. |
| + "prefer composites" nudge | gpt-5 | **16/20** — usage guidance made it *worse* → it's a tool-*quality* problem, not usage. |
| Failure-driven gen | gpt-5 | 5 real failures → 3 tools → **fixed task 63 (0→1)**; first clean fix. |
| Full hard set | gpt-5 | 29/36 vs base 31/36 = **net −2** (fixes 1, breaks ~3). |
| Weak-agent split | gpt-4o-mini agent / gpt-5 gen | base 34 → aug 33 = **net −1**; fixed 5, broke 6, +22% steps. |
| Net-benefit filter | gpt-4o-mini | one tool scored **net +3** and "survived"… |
| Multi-trial verdict | gpt-4o-mini | …but 50 tasks × 5 trials shows that tool is **net −6.4 pts, 0 robustly-improved tasks**. |

**Two walls:**
1. **Noise wall (the killer).** With a weak agent (temp 0.2) + a stochastic simulated user, the *same library*
   scores ±16% run-to-run (37 vs 29 / 50 for byte-identical config). Every effect we ever measured (−1, −2, +3)
   is smaller than the ruler's error → single-trial selection **selects noise**.
2. **Fix ≈ break.** Failure-driven tools genuinely fix real failures but break ~as many previously-passing
   tasks → net ≈ 0. The tools aren't wrong, they're *unselective*.

**Root cause = the two scores in the repo are both bad:**
- Runner's validation score (`experiments/libgen/runner.py:_validation_phase`): keep a tool iff it was
  *called without erroring* — **reward-blind**.
- Filter's score (`experiments/run_netbenefit_filter.py`): single-trial `fixed − broke` — **noise-dominated**.

---

# The Plan (v2): Scoring-Centric

**The bet:** every failure traces to *how we score a composite*. τ-bench reward is
`data_hash == gt_data_hash` — a **deterministic function of the final DB state, not of which tools were
called** (`tau_bench/envs/base.py:calculate_reward`). So we can score a tool by **rewriting a real trajectory
to use it and replaying it offline** — deterministic, cheap, no live agent, no simulated user, **no noise.**

**Structural payoff:** the offline replay score is **agent-model-independent** (reward depends only on env
writes) → the scoring gauntlet runs **once per candidate tool**, not per model. The model band only affects
which failures exist and live adoption.

## Reframe — decouple two questions

| Question | Nature | Scored by |
|---|---|---|
| **Correctness** — if used right in place of the pattern, does the task still pass / get fixed? | deterministic, cheap, model-independent | offline replay (S1–S3, S7) |
| **Adoption** — will the live agent pick it + call it with the right args? | stochastic, expensive, model-dependent | live multi-trial (S5) |

Offline correctness is a **necessary condition** and a perfect filter for the "subtly-wrong tool" mode. Live
runs move from *selection* (noisy, wasteful) to a final *confirmation* on a tiny shortlist.

## Scoring functions (pluggable registry — this is the part to revise)

All scorers share one interface so we can add/drop freely (`scoring/base.py`, to build in Phase 0):

```python
class Scorer:
    name: str
    kind: str   # "gate" (deterministic) | "confirm" (live) | "prior" (cheap tiebreak)
    def score(self, tool, trajs, env_factory) -> {value, keep: bool, detail}
REGISTRY = [S1, S2, S3, S4, S5, S6, S7]
```

**Deterministic gates (offline, cheap, model-independent)**

- **Substrate = GOLD trajectories, not agent trajectories.** *(Phase 0 finding: the stored agent `traj` is a
  message display that is truncated for final write actions — e.g. task 2 passed live but its `traj` contains no
  write at all. Gold actions (`info.task.actions`) are complete and passing by construction.)* Harness validated
  on gold: deterministic, reproduces pass 47/47, detects an altered write 46/47.
- **S1 — Substitution correctness (rewrite-and-replay). Two variants by tool type.** *(Grounding, Phase 0:
  reward = DB-hash, so it moves only for WRITE tools; most generated tools are READ-only.)*
  - **S1w — reward-replay (WRITE composites).** Splice the composite into the trajectory → `env.reset` →
    replay via `env.step` → `calculate_reward`. **Keep [REVISE]:** reward preserved on passing trajectories /
    flips 0→1 on the targeted failing one. Fully discriminative (the tool mutates `env.data`).
  - **S1r — return-equivalence (READ composites).** Reward can't move (reads don't touch `env.data`), so
    instead replay the read sub-sequence, capture its result, run the composite on the same env state, and
    assert it returns the **same/correct** value (e.g. same `order_id`). Deterministic correctness w/o reward.
  - Both emit per-task **step reduction** (→ S7).
  - **Run over BOTH trajectory populations:** *failing* trajectories test the FIX side (benefit; → S2) and
    *passing* trajectories test the REGRESSION side (do-no-harm; → S3). Deterministic net = fixes − regressions.
    For READ tools on *failing* trajectories, check the return against a **ground-truth target** from the gold
    actions (`info.task.actions`), since there's no reward to flip.
  - **Insight to carry into the paper:** a READ tool's live fix/break is pure **adoption** (behavioral) and is
    *unscoreable offline via reward* — consistent with the read-only survivor washing to noise in multi-trial.
    **WRITE tools are where deterministic offline selection has teeth.**
- **S2 — Counterfactual fix (rewrite-and-replay, FIX mode).** *Primary gate for failure-driven tools.* For each
  **failing** trajectory: strong model rewrites it inserting the composite (**k_rewrite [REVISE: 3]** attempts);
  replay each; **keep if ≥1 flips reward 0→1** — the tool *provably can* fix that failure before any live run.
- **S3 — Deterministic regression.** Replay the tool across held-out **already-passing** trajectories that don't
  need it. **Keep if zero reward drops.** S2 − S3 = **noise-free net benefit** (the honest replacement for the
  filter that selected noise).
- **S4 — Pattern incidence.** *Grep-cheap pre-filter.* How often the tool's target sub-pattern actually appears.
  **Drop if 0** (no adoption ceiling). Emits `adoption_ceiling`.
- **S7 — Step-efficiency (deterministic).** From S1's replay: mean calls removed/task. The **only** metric that
  works for reward-saturated strong models (gpt-5 solves retail 20/20). **Keep-rule [REVISE]:** `steps_saved > 0`.

**Live confirmation (stochastic, expensive — shortlist only)**

- **S5 — Multi-trial live net.** Base vs base+tool live, **k_trials [REVISE: 5]**, robust `fixed − broke` (sign
  stable in ≥⅗ trials) against the measured variance budget. **Keep if robust net > 0.** Headline-grade evidence
  of *adoption*.

**Cheap prior (tiebreak only)**

- **S6 — LLM-judge quality prior.** well-scoped? correctly typed? non-redundant? Rank ties only; never a sole gate.

*Extension scorers (generalization / arg-robustness / redundancy replays) are **deferred** — not in scope for
the first run. Parked here so we don't lose the idea.*

## Scoring pipeline (cheap → expensive)

```
candidate tool
  └─ S4  pattern incidence        (instant)     drop if 0
  └─ S1  substitution reward      (det, cheap)  gate: 100% preserve    [solved-traj tools]
  └─ S2  counterfactual fix       (det, cheap)  gate: >=1 provable fix  [failure tools]
  └─ S3  deterministic regression (det, cheap)  gate: 0 replay regressions
  └─ S7  step-efficiency          (det, free)   report; gate for gpt-5 arm
  └─ (S6 prior, advisory only)
        ──> OFFLINE SURVIVORS (small, high-quality)
  └─ S5  multi-trial live net     (live, exp)   confirm: robust net > 0
        ──> SELECTED LIBRARY
```

The noisy/expensive live step only runs on pre-vetted tools → kills both the **noise wall** and the **cost wall**.

## Results layout (deterministic vs live kept separate)

Deterministic-gate outputs and live-confirmation outputs are **never written to the same file or folder**, so a
noisy live number can never contaminate a reproducible offline one. Each scorer's `kind` decides where it writes:

```
experiments/scoring/
  offline/                       # DETERMINISTIC gates ONLY  (S1–S4, S7)
    <tool_name>.json             #   one file per tool: {S1:{...}, S2:{...}, S3, S4, S7, ...}, keep: bool
    summary.json                 #   rollup + offline shortlist (tools passing all enabled gates)
  live/                          # STOCHASTIC ONLY  (S5 + variance budget)
    variance/report_<model>_temp<t>.json
    <model>/<tool_name>.json     #   k-trial fixed/broke, per-trial rewards, robust net
    summary.json                 #   live-confirmed selected library
  priors/                        # S6 LLM-judge (noisy, advisory) — kept out of both above
    <tool_name>.json
```

Rules:
- **Offline files carry NO trial dimension** (deterministic → one value; re-running overwrites identically). Any
  file under `offline/` must be reproducible byte-for-byte.
- **Live files ALWAYS carry** `n_trials`, per-trial rewards, and the variance budget they were judged against.
- The **only bridge** is `offline/summary.json` → the shortlist Phase 3 live-confirms; Phase 3 *reads* it but
  writes only under `live/`. Nothing writes across the boundary.

## Execution phases (with gates)

- **Phase 0 — Infra & harness (~1 day).** Build `scoring/replay.py` (`reset → step → calculate_reward`, composites
  executed against `env.data`) + the S1–S4/S7 registry. Add **sweep-level resume** (skip arms whose `report_*.json`
  exists) and **auth resilience** (retry `ClientAuthenticationError` + credential refresh in `tau_bench/trapi_infer.py`;
  fixes the ~1h token-expiry crash). **GATE G0 (PoC), reflecting the read/write split:**
  (a) survivor `find_order_by_item_with_tracking` (READ) → **S1r return-equivalence** on its 7 fixed + 4 broke tasks;
  (b) a WRITE tool `authenticate_and_update_address_plus_cheapest_product` → **S1w reward-replay** on its fixed/broke
  tasks. Each deterministic score must cleanly explain the flips it *can* explain (return-value for read, reward for
  write). This validates the scorer split before any sweep. If replay disagrees with reality, the harness is wrong —
  fix before proceeding.
- **Phase 1 — Candidate generation across the model band (~1–2 days).** Generator stays **gpt-5** (via
  `LIBGEN_GEN_DEPLOYMENT`) regardless of agent.
  - **Primary agent: `gpt-4o`** — the untested capability band and the core bet: competent enough to *use* a
    composite without misfiring, still fallible enough to have failures worth fixing, and more *consistent* than
    mini (→ shrinks the ±16% noise that only bites the live step, S5). **[REVISE]** deployment string TBC
    (`TRAPI_DEPLOYMENT_NAME`, e.g. `gpt-4o_2024-11-20` — confirm what's live before running).
    - **GATE G1-headroom:** run gpt-4o base first (~20 min). Keep as primary only if base ∈ **~30–42/50**
      (real accuracy headroom). If it's near-saturated (~47+/50) it has the gpt-5 problem → fall back to mini as
      primary or introduce a harder task set. **[REVISE]** band bounds.
  - **Comparison endpoint: `gpt-4o-mini`** — the "clear headroom but noisy" data point (~34/50, 0.708
    multi-trial). Keeping it makes the capability-band result a *finding*, not a single anecdote.
  - Optional later: `gpt-4.1 / o4-mini`, `gpt-5` (saturated — step-efficiency axis only).
  - Per agent: base run → find its real failures → failure-driven generation → candidates (tagged with
    provenance). Carry the existing solved-trajectory candidates too.
- **Phase 2 — Offline scoring gauntlet (~hours).** Run the §pipeline over all candidates once each → ranked,
  **noise-free** shortlist.
- **Phase 3 — Live multi-trial confirmation (~1 day, shortlist only).** First measure the **variance budget**
  (same base library as two arms, k trials, temp 0.2 and 0.0, per model) so effects have an error bar; then S5 on
  survivors. **GATE G3:** ≥1 tool with robust live net > 0 on some model → positive result.
- **Phase 4 — Final A/B + write-up.** Selected library vs base, k=5–8, on the winning (model, library), variance
  budget as error bars. Headline = deterministic offline net + live-confirmed robust net + step savings, per band.

## Success criteria

- **Best case:** "self-generated tools help agents in a **capability band**, recovered by a **deterministic,
  replay-based selection score** that sidesteps ±16% live-eval noise" — a positive result **plus** a reusable method.
- **Method contribution (holds even if net stays flat):** the offline replay score decouples correctness from
  adoption, is model-independent, and demonstrates that single-trial live selection provably selects noise.

## Knobs to revise (summary)

Scorers + keep-rules · model band · k_rewrite / k_trials / N tasks / temps / variance target ·
preserve threshold (100% vs allow 1 miss) · train/val/test splits for generation vs scoring vs confirmation.

## First action once approved

Build `scoring/replay.py` and run the **G0 PoC** on the survivor's 11 known tasks — validates the whole approach on
existing data for ~free, before any sweep. Then Phase 1.

---

## What The Code Is Doing

At a high level, the active workflow is:

1. Load baseline trajectories from `final_results/`.
2. Split task IDs into train, validation, and test ranges.
3. Use the training trajectories to suggest a new composite function.
4. Generate a function definition with an LLM.
5. Run the training tasks again with the new function exposed through an MCP server.
6. Check whether the new function was actually used and whether it caused errors. **← NOTE: this is the
   reward-blind "score" the v2 plan replaces (S1–S5). Keeping the description for reference.**
7. If needed, correct the function using the failing trajectory.
8. Validate candidate functions on held-out tasks.
9. Copy only the validated functions into the next MCP server snapshot.
10. Run the test split using the final accepted tools.

The important thing to understand is that the repository is not learning a policy from scratch. It is trying to improve the tool library so that the agent can solve tasks with fewer steps and fewer repeated calls.

## Main Execution Path

> **UPDATED (the previous description was stale).** The active runtime is **config-driven**, not the old
> `libgen_experiment.py` → `run.py` shell-out.

The current entrypoint is:

1. `python libgen_experiment.py --config configs/libgen/retail.json` orchestrates the experiment.
2. It dispatches into [experiments/libgen/runner.py](experiments/libgen/runner.py), which drives the
   generation / validation / test phases.
3. Task execution goes through [tau_bench/run.py](tau_bench/run.py) (`tau_run`), which loads the environment,
   creates the agent, and runs tasks in parallel (`ThreadPoolExecutor`).
4. Inference is via [tau_bench/trapi_infer.py](tau_bench/trapi_infer.py): `completion()` = the **agent** model
   (env-driven `TRAPI_*`); `gen_completion()` = a **separate strong** model that only *designs* tools
   (`LIBGEN_GEN_*`, default gpt-5). Tool bodies are plain Python at runtime.
5. Trajectories are written to JSON checkpoint files and read back by the drivers.

The standalone `experiments/run_*.py` scripts are self-contained drivers for specific experiments (A/B, weak,
failure-driven, netbenefit, multitrial). The MCP server files under [mcp/](mcp/) are the evolving tool library
the agent sees.

## Training / Validation / Test Setup

The current split used in the experiment code is:

- Train: task IDs `0-49`
- Validation: task IDs `50-79`
- Test: task IDs `80-99`

The experiment driver uses training chunks to generate and correct candidate functions, then filters those functions on validation before testing them. **[REVISE]** the v2 plan may repurpose these splits: generation vs offline-scoring vs live-confirmation should use disjoint task sets to avoid overfitting.

## Relevant Files

These are the files a new contributor should read first.

### Experiment orchestration

- [libgen_experiment.py](libgen_experiment.py) - main experiment driver (config-driven; see Main Execution Path).
- [experiments/libgen/runner.py](experiments/libgen/runner.py) - generation / validation / test phase driver.
- [experiments/](experiments/) - self-contained `run_*.py` experiment drivers + their result folders.

### Tool generation logic

- [lib_gen.py](lib_gen.py) - LLM-backed function suggestion, definition, docstring repair, correction, and
  `get_new_func_from_failure` (failure-driven synthesis).
- [libgen_utils.py](libgen_utils.py) - metrics, trajectory inspection, MCP tool extraction, and server composition.
- [llmagent.py](llmagent.py) - shared LLM wrapper used by the generation agents.

### Benchmark runtime

- [tau_bench/run.py](tau_bench/run.py) - benchmark executor and agent factory (`tau_run`, parallel exec).
- [tau_bench/trapi_infer.py](tau_bench/trapi_infer.py) - TRAPI inference: agent `completion()` + strong `gen_completion()`.
- [tau_bench/types.py](tau_bench/types.py) - config and result types (`RunConfig`, `Action`, `Task`, `RewardResult`).
- [tau_bench/envs/](tau_bench/envs/) - environment definitions for retail and airline; `base.py` holds
  `calculate_reward` (the deterministic DB-hash reward the v2 scoring relies on).
- [tau_bench/agents/](tau_bench/agents/) - the various agent strategies.

### MCP server seed and snapshots

- [mcp/retail_server.py](mcp/retail_server.py) - base retail tool server (23 tools) used as the starting library.
- [mcp/airline_server.py](mcp/airline_server.py) - airline counterpart.
- [mcp/retail_server_initial.py](mcp/retail_server_initial.py) - initial snapshot used during experimentation.

### Data and benchmark artifacts

- [final_results/](final_results/) - baseline trajectories and experiment summaries.
- [few_shot_data/](few_shot_data/) - few-shot examples for benchmark agents.
- [fault_analysis.json](fault_analysis.json) and [fault_analysis_airline.json](fault_analysis_airline.json) - failure analysis inputs for the error-correcting workflow.
- [one_shot_retail_train.json](one_shot_retail_train.json) and [one_shot_retail_test.json](one_shot_retail_test.json) - one-shot data used by benchmark modes.

## Potentially Not Relevant

These files and folders may be useful for analysis or historical reference, but they do not appear to be part of the main library-generation workflow.

### Generated experiment outputs

- `libgen_experiment_output*/`
- `libgen_from_error_experiment_output*/`
- `improved_logs/`
- `results.json`
- `final_results/*.csv`
- `final_results/output.txt`

These are useful for analysis and reproducibility, but they are not source code.

### Generated or intermediate MCP snapshots

- Many of the versioned files under [mcp/](mcp/), such as `retail_server_after_generation_iteration_*`, `retail_server_after_validation_iteration_*`, `retail_server_before_generation_iteration_*`, `retail_server_temp_iteration_*`, `retail_server_test.py`, and `retail_server_test2.py`.
- Most files under [mcp2/](mcp2/) for the same reason.

These are mostly checkpoints of the evolving tool library, not hand-maintained code.

### Analysis and notebook material

- [analysis/](analysis/)
- [analysis.ipynb](analysis.ipynb)
- [intent_formalization.ipynb](intent_formalization.ipynb)
- [read_tasks.ipynb](read_tasks.ipynb)
- [auto_error_analysis/](auto_error_analysis/)
- [llm_as_a_judge/](llm_as_a_judge/)
- [post_condition/](post_condition/)
- [temp.py](temp.py)

These are likely research or diagnostic folders, not part of the main day-to-day workflow.

### Legacy or alternate experiments

- [error_correcting_libgen_experiment.py](error_correcting_libgen_experiment.py)
- [auto_error_identification.py](auto_error_identification.py)
- [combine_results.py](combine_results.py)
- [libgen_experiment_output_trial_2/](libgen_experiment_output_trial_2/) through `_trial_7/`

These look like experiment branches, not the canonical runtime path.

## Suggested Working Model For The Project

- `tau_bench/` is the benchmark runtime.
- `mcp/` is the evolving tool library that the agent sees.
- `libgen_experiment.py` + `experiments/libgen/runner.py` are the experiment controller.
- `lib_gen.py` decides what function to propose and how to rewrite it.
- `libgen_utils.py` checks whether the new tool was actually used and whether it failed.
- `scoring/` (to build) holds the v2 deterministic replay scorers.
- `final_results/` contains the trajectories used to bootstrap the process.

## Short Summary

This repo is an experimental system for mining composite tools from agent trajectories. The generation core is
small and works; the open problem is **scoring** — prior scores were reward-blind or noise-dominated, which is
why mined tools never showed net benefit. The v2 plan above replaces scoring with **deterministic
rewrite-and-replay** (decoupling tool correctness from adoption) plus multi-model, multi-trial confirmation. A
contributor should read “Where We Are Now” and “The Plan (v2)” first, then the files in “Relevant Files.”
