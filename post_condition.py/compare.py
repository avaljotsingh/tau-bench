from analysis.code import Code
import json 


def get_original_plan(task):
    new_lines = []
    lines = task[1]['content'].splitlines()
    for i, line in enumerate(lines):
        if line.startswith("agent-plan:```"):
            index = i
            break
    for i in range(index + 1, len(lines)):
        if lines[i].endswith("```"):
            break
        new_lines.append(lines[i])
    return Code('\n'.join(new_lines))

def get_new_plan(task):
    new_lines = []
    lines = task[2]['content'].splitlines()
    for i, line in enumerate(lines):
        if line.startswith("```"):
            index = i
            break
    for i in range(index + 1, len(lines)):
        if lines[i].endswith("```"):
            break
        new_lines.append(lines[i])
    return Code('\n'.join(new_lines))


file_path = 'output.json'
data = json.load(open(file_path))
tasks = data
task = tasks[1]
# print(type(task))
# print(task.keys())

original_plan = get_original_plan(task)
new_plan = get_new_plan(task)
original_plan.pretty_print(False)
new_plan.pretty_print(False)
# results = data.analyze_codes_static()
# print_results(results)
# data.modify_codes_static()
# data.print_generated_plan(0)
# data.print_generated_plan_with_post(0)
# data.analyze_codes_dynamic()
# data.print_generated_plan(0)
# data.print_ground_truth_plan(0)