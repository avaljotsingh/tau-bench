from tau_bench.trapi_infer import completion
from tau_bench.trapi_infer import model_dump
import json
from analysis.code import Code

SYTEM_PROMPT = '''
Hello. You are a verifier. Given the user intent, a web agent genertated a program (plan) to fulfill the intent.
However, there may be some problems with the generated plan. 
For each step in the plan, generate postconditions in the form of assert statements to check if the step executed correctly and gets the desired output.
Output a new plan without correctly inserted assert statements after each step.
Here is the conversation of the agent with the user
'''

def get_instructions(task):
    return task['traj'][0]['content']

def get_intent(task):
    return task['traj'][1]['content']

def get_plan(task):
    return task['traj'][2]['content']

def get_prompt(task):
    p1 = get_instructions(task)
    p2 = get_intent(task)
    p3 = get_plan(task)
    content = f"user-instruction:{p1}\nuser-intent:{p2}\nagent-plan:{p3}"
    messages = [
        {"role": "system", "content": SYTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages

def get_prompt_old(task):
    p1 = get_instructions(task)
    p2 = get_intent(task)
    p3 = get_plan(task)
    messages = [
        {"role": "system", "content": SYTEM_PROMPT},
        {"role": "user-instruction", "content": p1},
        {"role": "user-intent", "content": p2},
        {"role": "agent-plan", "content": p3},
    ]
    return messages

def main(data):
    tasks = data
    results = []
    for i, task in enumerate(tasks):
        print(i)
        prompt = get_prompt(task)
        response = completion(messages=prompt)
        result = model_dump(response.choices[0].message)
        prompt.append(result)
        results.append(prompt)
    with open('output.json', 'w') as f:
        json.dump(results, f, indent=4)

filepath = 'one-shot-gpt-4o-0.0_range_0--1_user-gpt-4o-one-shot_0609102847.json'
data = json.load(open(filepath))
main(data)