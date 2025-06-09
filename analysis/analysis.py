import json
from termcolor import colored

from code import Code
from static_checker import StaticChecker
from execute_plan import PlanExecutor
from deepdiff import DeepDiff
from tabulate import tabulate



class Plan:
    def exec_plan(plan):
        executor = PlanExecutor(plan)
        executor.execute()
        final_data = executor.get_data()
        return final_data
    
class Task:
    def __init__(self, task):
        self.task = task
        self.ground_truth_plan = self.get_ground_truth_plan()
        self.generated_plan = self.get_generated_plan()
        self.intent = self.get_intent()

    def get_ground_truth_plan(self):
        actions = self.task['info']['task']['actions']
        lines= []
        for action in actions:
            func_name = action['name']
            kwargs = action['kwargs']
            args = [f'{k}="{v}"' for k, v in kwargs.items()]
            func_call = f'{func_name}({", ".join(args)})'
            lines.append(func_call)
        plan = Code('\n'.join(lines))
        return plan

    def get_generated_plan(self):
        return Code(self.task['traj'][2]['content'])
    
    def get_intent(self):
        return self.task['traj'][1]['content']

    def exec_ground_truth_plan(self):
        return Plan.exec_plan(self.ground_truth_plan)

    def execute_generated_plan(self):
        return Plan.exec_plan(self.generated_plan)
    
    def check_static(self, flags=['syntax', 'check_dead_code', 'undefined_vars', 'type_check']):
        plan = self.generated_plan
        static_checker = StaticChecker(plan)
        return static_checker.analyze(flags)

    def dead_code_check(self):
        plan = self.generated_plan
        static_checker = StaticChecker(plan)
        flags = ['remove_dead_code']
        return static_checker.analyze(flags)

    def check_correctness(self):
        final_ground_truth_data = self.exec_ground_truth_plan()
        final_generated_data = self.execute_generated_plan()
        diff = DeepDiff(final_ground_truth_data, final_generated_data, verbose_level=2)
        return diff
    
class Data:
    def __init__(self, filename):
        self.data = json.load(open(file_path))

    def get_task(self, i):
        return Task(self.data[i])

    def analyze_codes_dynamic(self):
        diffs = {}
        for i in range(len(self.data)):
            diffs[i] = self.get_task(i).check_correctness()
        return diffs

    def analyze_codes_static(self):
        results = {}
        for i in range(len(self.data)):
            task = self.get_task(i)
            results[i] = task.check_static()
        return results
    
    def print_plan(self, i):
        task = self.get_task(i)
        plan = task.generated_plan
        plan.pretty_print(False)


def to_symbol_val(value):
    return '✔' if value else '✘'

def to_symbol(res):
    flag = to_symbol_val(res[0])
    if res[1]:
        return f"{flag} ({res[1]})"
    else:
        return flag


def print_results(results):
    table = []
    for index, checks in results.items():
        row = {
            "Index": index,
            "Syntax": to_symbol(checks['syntax']),
            "Dead Code": to_symbol(checks['check_dead_code']),
            "Undefined Vars": to_symbol(checks['undefined_vars']),
            "Type Check": to_symbol(checks['type_check']),
        }
        table.append(row)
    print(tabulate(table, headers="keys", tablefmt="grid"))


file_path = 'one-shot-gpt-4o-0.0_range_0--1_user-gpt-4o-one-shot_0609102847.json'
data = Data(file_path)
results = data.analyze_codes_static()
print_results(results)