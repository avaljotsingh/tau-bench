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

## What The Code Is Doing

At a high level, the active workflow is:

1. Load baseline trajectories from `final_results/`.
2. Split task IDs into train, validation, and test ranges.
3. Use the training trajectories to suggest a new composite function.
4. Generate a function definition with an LLM.
5. Run the training tasks again with the new function exposed through an MCP server.
6. Check whether the new function was actually used and whether it caused errors.
7. If needed, correct the function using the failing trajectory.
8. Validate candidate functions on held-out tasks.
9. Copy only the validated functions into the next MCP server snapshot.
10. Run the test split using the final accepted tools.

The important thing to understand is that the repository is not learning a policy from scratch. It is trying to improve the tool library so that the agent can solve tasks with fewer steps and fewer repeated calls.

Example pattern:

- Existing tools: `get_user_details`, `get_order_details`
- User wants: return shoes
- Current behavior: get user details, inspect orders, then call order details multiple times
- Better composite tool: `find_correct_order`

That is the kind of abstraction the system is trying to discover from trajectories.

## Main Execution Path

The active entrypoint for the library-generation experiment is [libgen_experiment.py](libgen_experiment.py).

The runtime path is:

1. [libgen_experiment.py](libgen_experiment.py) orchestrates the experiment.
2. It shells out to [run.py](run.py).
3. [run.py](run.py) forwards into [tau_bench/run.py](tau_bench/run.py).
4. [tau_bench/run.py](tau_bench/run.py) loads the chosen environment, creates the agent, and executes tasks.
5. The resulting trajectories are written to JSON checkpoint files and read back by the experiment driver.

The MCP server files under [mcp/](mcp/) are the evolving tool library that the agent sees during each iteration.

## Training / Validation / Test Setup

The current split used in the experiment code is:

- Train: task IDs `0-49`
- Validation: task IDs `50-79`
- Test: task IDs `80-99`

The experiment driver uses training chunks to generate and correct candidate functions, then filters those functions on validation before testing them.

The generated run artifacts are written into folders such as:

- `libgen_experiment_output/`
- `libgen_experiment_output_trial_2/`
- `libgen_experiment_output_trial_3/`
- `libgen_experiment_output_trial_4/`
- `libgen_experiment_output_trial_5/`
- `libgen_experiment_output_trial_6/`
- `libgen_experiment_output_trial_7/`
- `libgen_from_error_experiment_output_trial_1/`
- `libgen_from_error_experiment_output_trial_2/`
- `improved_logs/`

These directories are important as experiment outputs, but they are not the core source code.

## Relevant Files

These are the files a new contributor should read first.

### Experiment orchestration

- [libgen_experiment.py](libgen_experiment.py) - main experiment driver for generation, validation, and testing.
- [error_correcting_libgen_experiment.py](error_correcting_libgen_experiment.py) - alternate experiment flow that uses failure analysis.

### Tool generation logic

- [lib_gen.py](lib_gen.py) - LLM-backed function suggestion, function definition, docstring repair, and correction.
- [libgen_utils.py](libgen_utils.py) - metrics, trajectory inspection, MCP tool extraction, and server file composition.
- [llmagent.py](llmagent.py) - shared LLM wrapper used by the generation agents.

### Benchmark runtime

- [run.py](run.py) - top-level CLI wrapper for the benchmark.
- [tau_bench/run.py](tau_bench/run.py) - benchmark executor and agent factory.
- [tau_bench/types.py](tau_bench/types.py) - config and result types used by the runtime.
- [tau_bench/envs/](tau_bench/envs/) - environment definitions for retail and airline.
- [tau_bench/agents/](tau_bench/agents/) - the various agent strategies.

### MCP server seed and snapshots

- [mcp/retail_server.py](mcp/retail_server.py) - base retail tool server used as the starting library.
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
- [libgen_experiment_output_trial_2/](libgen_experiment_output_trial_2/)
- [libgen_experiment_output_trial_3/](libgen_experiment_output_trial_3/)
- [libgen_experiment_output_trial_4/](libgen_experiment_output_trial_4/)
- [libgen_experiment_output_trial_5/](libgen_experiment_output_trial_5/)
- [libgen_experiment_output_trial_6/](libgen_experiment_output_trial_6/)
- [libgen_experiment_output_trial_7/](libgen_experiment_output_trial_7/)

These look like experiment branches, not the canonical runtime path.

## Suggested Working Model For The Project

- `tau_bench/` is the benchmark runtime.
- `mcp/` is the evolving tool library that the agent sees.
- `libgen_experiment.py` is the experiment controller.
- `lib_gen.py` decides what function to propose and how to rewrite it.
- `libgen_utils.py` checks whether the new tool was actually used and whether it failed.
- `final_results/` contains the trajectories used to bootstrap the process.

## Short Summary

This repo is an experimental system for mining composite tools from agent trajectories. The core workflow is fairly small, but it is surrounded by many generated outputs, alternate experiments, and analysis artifacts. A new contributor should focus on the files listed in the “Relevant Files” section first and treat the rest as supporting data or archive material until proven otherwise.