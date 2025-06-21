import json

from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.utils import *
from tau_bench.globals import *

PRECONDITION_SYSTEM_PROMPT = f'''
Hi. You are an expert at doing sanity checks.
Here is a conversation between a customer and a support. 
I am also giving you the actions taken so far and the corresponding environment response.
The support has now genarated the next action. 
But the action may be risky or have side-effects. 
Can you generate some executable preconditions that must be checked before excecuting the action?
Please generate an executable code for the prechecks. The tools that you can use are as follows.
Do not output any explanation. Only generate an executable code. Do not define any new functions. 
I have given the python code for the actions executed so far. And also their outputs in the following comments.
In your precondition code, you can only use the variables that are defined in the python code.
The latest action has not been executed yet. So, the corresponding variable is nto defined yet and the output is unknown. 
Do not hallucinate the values of the variables.
You also need to check that in the new proposed function call, are the arguments are correct and are of the correct type?
You do not need to check the existence of the function or if it is callable. We will make sure before execution that it is callable.
You don't need to add any print statement. 
Raise an exception for each error so that I can run the code and catch all the exceptions.
'''

class PreconditionAgent():
    def __init__(self, tools_info):
        self.tools_info = tools_info

    def generate_context(self, records, action):
        context = [{"role": "system", "content": PRECONDITION_SYSTEM_PROMPT}]
        user_message = ''
        for i in range(len(records.conversation)):
            role = records.conversation[i]["role"]
            index = records.conversation[i]['index']
            if role == Role.USER:
                user_message += f"customer: {records.user_messages[index]}\n"
            elif role == Role.ENV:
                user_message += f"# {records.env_messages[index]}\n```\n"
            elif role == Role.ACTION_AGENT:
                user_message += f"support: {records.action_agent_messages[index]}\n"
            elif role == Role.TOOL_CALL:
                user_message += f"Action taken: \n```python \n{records.plan[index]}\n"
            elif role == Role.TOOL_OUTPUT:
                user_message += f"Action output: \n# {records.plan_outputs[index]}\n```\n"
            elif role == Role.PRECONDITION_AGENT:
                user_message += f"Precondition checked before: {records.preconditions[index]}\n"
            else:
                raise Exception(f'Role: {role} not defined')
        user_message += f"python code for the new action: \n```python\nvar_{records.var_counter} = {action.name}({(str(action.kwargs))[1:-1]})\n```\n"
        context.append({"role": "user", "content": user_message})
        return context
    
    def generate_precondition(self, records, action):
        start_time = time.time()
        context = self.generate_context(records, action)
        res = completion(messages=context, tools=self.tools_info)
        precondition = self.parse_precondition_message(res.choices[0].message)
        precondition_agent_time.record_time(time.time() - start_time)
        return precondition
    
    def parse_precondition_message(self, precondition):
        code = precondition['content']
        if code is not None:
            lines = code.split('\n')
            new_lines = []
            flag = False
            for line in lines:
                if flag:
                    if line.startswith("```"):
                        flag = False
                        break
                    new_lines.append(line)
                if line.startswith("```"):
                    flag = True
            code = '\n'.join(new_lines)
        else:
            code = '' 
        return code
    
    def custom_assert(self, condition, message):
        if not condition:
            raise AssertionError(message)
    
    def smart_parse_context(self, assignments):
        local_vars = {}
        for line in assignments:
            if '=' not in line:
                raise ValueError(f"Invalid assignment: {line}")
            key, value_str = map(str.strip, line.split('=', 1))
            if value_str == '':
                value = ''
            elif 'error' in value_str.lower():
                value = value_str
            elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value_str):
                value = value_str
            else:
                try:
                    value = json.loads(value_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to decode JSON for '{key}': {e}")
            local_vars[key] = value
        return local_vars
    
    def parse_context(self, assignments):
        local_vars = {}
        for line in assignments:
            exec(line, {}, local_vars)
        return local_vars


    def execute_precondition(self, precondition: str, records):
        start_time = time.time()
        errors = []
        context = self.smart_parse_context(records.plan_outputs)
        exec_env = {"__builtins__": __builtins__}
        exec_env.update(context)
        try:
            exec(precondition, exec_env)
        except Exception as e:
            errors.append(f"{e.__class__.__name__}: {e}")
        precondition_agent_time.record_time(time.time() - start_time)
        return errors
    