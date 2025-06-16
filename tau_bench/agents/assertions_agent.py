import json
import traceback
import ast
import re
from tau_bench.trapi_infer import completion, model_dump
from typing import List, Optional, Dict, Any
from enum import Enum, auto

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from termcolor import colored

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
You do not need to check the existence of the function.
You don't need to add any print statement. 
Raise an exception for each error so that I can run the code and catch all the exceptions.
'''


POSTCONDITION_SYSTEM_PROMPT = f'''
Hi. You are an expert at planning tasks.
Here is a list of tasks that represent the intentions of the customer. 
However, the in the recent conversation with a support agent, you may need to update the tasks to better reflect the customer's needs.
Following is the existing set of tasks, followed by the recent conversation between the customer and the support.
I have represented the actions taken by the support agent in python code.
Output only the new postconditions that you discover fronm the conversation.
Output in a numbered list in different lines.
Do no repeat any tasks that is already known.
'''
        

POSTCONDITION_CHECK_SYSTEM_PROMPT = f'''
Hi. You are an expert at planning tasks.
Here is the list of actions taken byu the support agent. 
Following is one of the tasks that represent the intentions of the customer. 
Do you think that the task is solved by the actions taken by the support agent.
Output only true of false.
'''





class Role(Enum):
    USER = auto()
    ENV = auto()
    ACTION_AGENT = auto()
    TOOL_CALL = auto()
    TOOL_OUTPUT = auto()
    PRECONDITION_AGENT = auto()
    PRECONDITION_OUTPUT = auto()
    POSTCONDITION_AGENT = auto()
    POSTCONDITION_OUTPUT = auto()
    

def print_message(role, message):
    if role==Role.USER:
        print(colored(f'User: \n{message}', 'blue'))
    elif role==Role.ENV:
        print(colored(f'Environment: \n{message}', 'cyan'))
    elif role==Role.ACTION_AGENT:
        print(colored(f'Action Agent: \n{message.kwargs["content"]}', 'green'))
    elif role==Role.TOOL_CALL:
        print(colored(f'Tool: \n{message}', 'yellow'))
    elif role==Role.TOOL_OUTPUT:
        print(colored(f'Tool: \n{message}', 'yellow'))
    elif role==Role.PRECONDITION_AGENT:
        print(colored(f'Precondition Agent: \n{message}', 'red'))
    elif role==Role.PRECONDITION_OUTPUT:
        print(colored(f'Precondition Output: \n{message}', 'red'))
    elif role==Role.POSTCONDITION_AGENT:
        print(colored(f'Postcondition Agent: \n{message}', 'magenta'))
    elif role==Role.POSTCONDITION_OUTPUT:
        print(colored(f'Postcondition Output: \n{message}', 'magenta'))
    else:
        print(role)
        raise f'Role: {role} not defined'
    
class Records():
    def __init__(self):
        self.conversation = []
        self.user_messages = []
        self.env_messages = []
        self.action_agent_messages = []
        self.plan = []
        self.plan_outputs = []
        self.preconditions = []
        self.preconditions_outputs = []
        self.postconditions = []
        self.postconditions_outputs = []
        self.var_counter = 0

class ActionAgent():
    def __init__(self, tools_info, wiki, env):
        self.tools_info = tools_info
        self.wiki = wiki
        self.env = env

    def generate_next_action_message(self, messages):
        res = completion(messages=messages, tools=self.tools_info)
        return res.choices[0].message
    
    def get_context(self, records: Records):
        context = [{"role": "system", "content": self.wiki}]
        for i in range(len(records.conversation)):
            role = records.conversation[i]["role"]
            index = records.conversation[i]['index']
            if role == Role.USER:
                context.append({"role": "user", "content": records.user_messages[index]})
            elif role == Role.ENV:
                context.append({"role": "user", "content": f'# {records.env_messages[index]}'})
            elif role == Role.ACTION_AGENT:
                context.append({"role": "assistant", "content": records.action_agent_messages[index]})
            elif role == Role.TOOL_CALL:
                context.append({"role": "assistant", "content": records.plan[index]})
            elif role == Role.TOOL_OUTPUT:
                context.append({"role": "assistant", "content": f'# {records.plan_outputs[index]}'})
            else:
                raise Exception(f'Role: {role} not defined')
        return context
    
    def generate_next_action(self, records):
        start_time = time.time()
        context = self.get_context(records)
        action_message = self.generate_next_action_message(context)
        action = self.message_to_action(action_message)
        action_agent_time.record_time(time.time() - start_time)
        return action
    
    
    def message_to_action(self, message: Dict[str, Any]) -> Action:
        if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
            tool_call = message["tool_calls"][0]
            return Action(
                name=tool_call["function"]["name"],
                kwargs=json.loads(tool_call["function"]["arguments"]),
            )
        else:
            return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
    
    def execute_action(self, action: Action, env: Env):
        start_time = time.time()
        env_response = env.step(action)
        env_time.record_time(time.time() - start_time)
        return env_response.done, env_response.observation, env_response.reward, env_response.info

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
    
    
class PostconditionAgent():
    def __init__(self, tools_info, postconditions=[]):
        self.tools_info = tools_info
        self.conversation_index = 0

    def parse_postcondition(self, postcondition_message):
        matching_lines = []
        for line in postcondition_message.splitlines():
            match = re.match(r'^\d+[\.\)\-\s]*', line)
            if match:
                cleaned_line = line[match.end():].strip()  
                matching_lines.append(cleaned_line)
        return matching_lines
    
    def get_message_from_records(self, records):
        message = ''
        for i in range(self.conversation_index, len(records.conversation)):
            role = records.conversation[i]['role']
            index = records.conversation[i]['index']
            if role == Role.USER:
                message += f"customer: {records.user_messages[index]}\n"
            if role == Role.ENV:
                message += f"# {records.env_messages[index]}\n```\n"
            elif role == Role.ACTION_AGENT:
                message += f"Support: {records.action_agent_messages[index]}\n"
            elif role == Role.TOOL_CALL:
                message += f"python code for action: \n```python \n{records.plan[index]}\n"
            elif role == Role.TOOL_OUTPUT:
                message += f"# {records.plan_outputs[index]}\n```\n"
        return message
    
    def get_context_for_generation(self, current_postconditions, message):
        context = [{"role": "system", "content": POSTCONDITION_SYSTEM_PROMPT}]
        message = f'tasks: {current_postconditions}\n' + message
        context.append({"role": "user", "content": message})
        return context
    
    def generate_new_postcondition(self, current_postconditions, message):
        context = self.get_context_for_generation(current_postconditions, message)
        res = completion(messages=context, tools=self.tools_info)
        new_post_condition_message = res.choices[0].message['content']
        new_post_conditions = self.parse_postcondition(new_post_condition_message)
        return new_post_conditions
    
    def check_postcondition(self, current_postconditions, message):
        tasks_solved = []
        for task in current_postconditions:
            task_user_message = message + f"\n task: {task}\n"
            messages = [{"role": "system", "content": POSTCONDITION_CHECK_SYSTEM_PROMPT}, {"role": "user", "content": task_user_message}]
            res = completion(messages=messages)
            check_passed = res.choices[0].message['content']
            tasks_solved.append('true' in check_passed or 'True' in check_passed)
        return tasks_solved
    
    def register_conversation(self, current_postconditions, records):
        start_time = time.time()
        message = self.get_message_from_records(records)
        self.conversation_index = len(records.conversation)
        tasks_solved = self.check_postcondition(current_postconditions, message)
        new_postconditions = self.generate_new_postcondition(current_postconditions, message)
        pending_postconditions = []
        for i in range(len(tasks_solved)):
            if not tasks_solved[i]:
                pending_postconditions.append(current_postconditions[i])
        postcondition_agent_time.record_time(time.time() - start_time)
        return pending_postconditions, new_postconditions
    

class AssertionsAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki

    def register_message(self, role, message):
        print_message(role, message)
        if role == Role.USER:
            self.records.conversation.append({'role': role, 'index': len(self.records.user_messages)})
            self.records.user_messages.append(message)
        elif role == Role.ENV:
            assert('error' in message or 'Error' in message)
            new_message = f'var_{self.records.var_counter-1} = "{message}"'
            self.records.conversation.append({'role': role, 'index': len(self.records.env_messages)})
            self.records.env_messages.append(new_message)
        elif role == Role.ACTION_AGENT:
            assert(message.name == RESPOND_ACTION_NAME)
            action = message
            self.records.conversation.append({'role': role, 'index': len(self.records.action_agent_messages)})
            self.records.action_agent_messages.append(action.kwargs['content'])
        elif role == Role.TOOL_CALL:
            self.records.conversation.append({'role': role, 'index': len(self.records.plan)})
            action = message
            name = action.name
            kwargs = action.kwargs
            arg_str = ', '.join(f"{k}='{v}'" for k, v in kwargs.items())
            function_call_str = f"var_{self.records.var_counter} = {name}({arg_str})"
            self.records.var_counter += 1
            self.records.plan.append(function_call_str)
        elif role == Role.TOOL_OUTPUT:
            new_message = f'var_{self.records.var_counter-1} = {message}'
            self.records.conversation.append({'role': role, 'index': len(self.records.plan_outputs)})
            self.records.plan_outputs.append(new_message)
        elif role == Role.PRECONDITION_AGENT:
            self.records.preconditions.append(message)
        elif role == Role.PRECONDITION_OUTPUT:
            self.records.preconditions_outputs.append(message)
        elif role == Role.POSTCONDITION_AGENT:
            self.records.postconditions.append(message)
        elif role == Role.POSTCONDITION_OUTPUT:
            self.records.postconditions_outputs.append(message)
        else:
            print(role)
            raise f'Role: {role} not defined'

    def register_output_from_action(self, action_message, observation):
        if action_message.name == RESPOND_ACTION_NAME:
            self.register_message(Role.USER, observation)
        else:
            if 'error' in observation or 'Error' in observation:
                self.register_message(Role.ENV, observation)
            else:
                self.register_message(Role.TOOL_OUTPUT, observation)
        

    def solve(self, env: Env, task_index: Optional[int] = None, max_actions: int = 30, max_refinements: int = 1) -> SolveResult:
        self.records = Records()
        self.action_agent = ActionAgent(tools_info=self.tools_info, wiki=self.wiki, env=env)
        self.precondition_agent = PreconditionAgent(tools_info=self.tools_info)
        self.postcondition_agent = PostconditionAgent(tools_info=self.tools_info)
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        self.register_message(Role.USER, obs)
        self.postconditions = []
        

        done = False
        action_iteration = 0
        while not done and action_iteration < max_actions:
            print(action_iteration)
            print_times()
            action_iteration += 1
            action = self.action_agent.generate_next_action(self.records)
            if action.name == RESPOND_ACTION_NAME:
                action_role = Role.ACTION_AGENT 
                precondition_passed = True
            else:
                action_role = Role.TOOL_CALL 
                precondition_passed = False
            refinement_index = 0
            while not precondition_passed and refinement_index < max_refinements:
                refinement_index += 1
                precondition = self.precondition_agent.generate_precondition(self.records, action)
                self.register_message(Role.PRECONDITION_AGENT, precondition)
                precondition_passed = self.precondition_agent.execute_precondition(precondition, self.records)
                self.register_message(Role.PRECONDITION_OUTPUT, precondition_passed)
            self.register_message(action_role, action)
            done, observation, reward, info = self.action_agent.execute_action(action, env)
            self.register_output_from_action(action, observation)
            old_postconditions, new_postconditions = self.postcondition_agent.register_conversation(self.postconditions, self.records)
            self.register_message(Role.POSTCONDITION_AGENT, old_postconditions)
            self.register_message(Role.POSTCONDITION_AGENT, new_postconditions)
            self.postconditions = old_postconditions + new_postconditions
        
        
        return SolveResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            records=self.records,
        )