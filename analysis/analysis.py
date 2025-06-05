import json
from termcolor import colored

from static_checker import StaticChecker
from code import Code

def analyse_codes(filepath):
    data = json.load(open(filepath))
    for i, task in enumerate(data):
        code = Code(get_plan(task))
        flags = ['syntax', 'check_dead_code', 'remove_dead_code']
        static_checker = StaticChecker(code)
        flag, comment = static_checker.check(flags)
        if not flag:
            print(f'Task {i}: ', colored(comment, 'red'))
        else:
            print(f'Task {i}: ', colored(comment, 'green'))
    

def get_intent(task):
    return task['traj'][1]['content']

def get_plan(task):
    return task['traj'][2]['content']

def print_code(task, line_numbers=False):
    code_str = Code(task['traj'][2]['content'])
    code_str.add_imports()
    code_str.pretty_print_code(line_numbers)


file_path = 'one-shot-gpt-4o-0.0_range_0--1_user-gpt-4o-one-shot_0603172050.json'
analyse_codes(file_path)
