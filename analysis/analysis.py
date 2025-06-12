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
        self.get_generated_plan_with_post = self.get_generated_plan_with_post()
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
    
    def get_generated_plan_with_post(self):
        return Code(self.task['traj'][3]['content'])
    
    def get_intent(self):
        return self.task['traj'][1]['content']

    def exec_ground_truth_plan(self):
        return Plan.exec_plan(self.ground_truth_plan)

    def execute_generated_plan(self):
        plan = self.generated_plan
        return Plan.exec_plan(plan)
    
    def check_static(self, flags=['syntax', 'check_dead_code', 'undefined_vars', 'type_check']):
        plan = self.generated_plan
        static_checker = StaticChecker(plan)
        return static_checker.analyze(flags)
    
    def modify_static(self, flags=['add_hash']):
        plan = self.generated_plan
        static_checker = StaticChecker(plan)
        return static_checker.modify(flags)

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
    
    def update_plan(self, plan):
        self.generated_plan = plan
    
class Data:
    def __init__(self, filename):
        self.data = json.load(open(file_path))
        self.tasks = [self.get_task(i) for i in range(len(self.data))]


    def get_task(self, i):
        return Task(self.data[i])

    def analyze_codes_dynamic(self):
        diffs = {}
        for i, task in enumerate(self.tasks):
            diffs[i] = task.check_correctness()
        return diffs

    def analyze_codes_static(self):
        results = {}
        for i, task in enumerate(self.tasks):
            results[i] = task.check_static()
        return results
    
    def modify_codes_static(self):
        for i, task in enumerate(self.tasks):
            new_plan = task.modify_static()
            # print(task.generated_plan)
            task.update_plan(new_plan)
            # print(task.generated_plan)
            # kdug
    
    def print_generated_plan(self, i, line_numbers=False):
        task = self.get_task(i)
        plan = task.generated_plan
        plan.remove_imports()
        plan.pretty_print(line_numbers)

    def print_generated_plan_with_post(self, i, line_numbers=False):
        task = self.get_task(i)
        plan = task.generated_plan_with_post
        plan.remove_imports()
        plan.pretty_print(line_numbers)

    def print_ground_truth_plan(self, i):
        task = self.get_task(i)
        plan = task.ground_truth_plan
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


file_path = 'one-shot-none-0.1_range_0--1_user-none-one-shot_0610114914.json'
data = Data(file_path)
# results = data.analyze_codes_static()
# print_results(results)
# data.modify_codes_static()
# data.print_generated_plan(0)
data.print_generated_plan_with_post(0)
data.analyze_codes_dynamic()
# data.print_generated_plan(0)
# data.print_ground_truth_plan(0)