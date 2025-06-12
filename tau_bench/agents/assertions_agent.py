import json
import traceback
import ast
import re
from tau_bench.trapi_infer import completion, model_dump
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from termcolor import colored

SYSTEM_PROMPT = ""

class AssertionsAgent(Agent):
    counter = 0
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
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.plan = []
        self.tool_call_codes = []
        self.variable_defs = []
        self.messages = [
            {"role": "system", "content": self.wiki},
        ]

    def generate_postcondition(self, task_index: Optional[int] = None):
        '''
        Generate the postcondition for the task.
        This is a predicate that must hold for the task to be solved.
        The propositions represent the subintents of the task.
        The proporitions are in natural language.
        The predicate is a formula in propositional logic.
        '''
        pass

    def register_precondition(self, precondition: Action):
        self.plan.append((0, precondition))
    
    def register_action(self, action: Action):
        self.plan.append((1, action))

    def add_action_and_response(self, action_message, action, env_observation):
        if action.name != RESPOND_ACTION_NAME:
            action_message["tool_calls"] = action_message["tool_calls"][:1]
            self.messages.extend(
                [
                    action_message,
                    {
                        "role": "tool",
                        "tool_call_id": action_message["tool_calls"][0]["id"],
                        "name": action_message["tool_calls"][0]["function"]["name"],
                        "content": env_observation,
                    },
                ]
            )
        else:
            self.messages.extend(
                [
                    action_message,
                    {"role": "user", "content": env_observation},
                ]
            )

    def add_precondition_and_response(self, precondition_message, precondition, env_response):
        if precondition.name != RESPOND_ACTION_NAME:
            precondition_message["tool_calls"] = precondition_message["tool_calls"][:1]
            self.messages.extend(
                [
                    precondition_message,
                    {
                        "role": "tool",
                        "tool_call_id": precondition_message["tool_calls"][0]["id"],
                        "name": precondition_message["tool_calls"][0]["function"]["name"],
                        "content": env_response.observation,
                    },
                ]
            )
        else:
            self.messages.extend(
                [
                    precondition_message,
                    {"role": "user", "content": env_response.observation},
                ]
            )
    
    def check_postcondition(self):
        '''
        Given the plan executed so far, check what part of the postcondition has been solved.
        Return a tasks that still need to be solved.
        Think how to do this. 
        '''
        pass

    def execute_action(self, env: Env, action: Action):
        env_response = env.step(action)
        if action.name != RESPOND_ACTION_NAME:
            
            self.tool_call_codes.append(
                f'var_{AssertionsAgent.counter} = {action.name}({(json.dumps(action.kwargs))[1:-1]})'
            )
            self.variable_defs.append(
                f'var_{AssertionsAgent.counter} = {env_response.observation}'
            )
            AssertionsAgent.counter += 1
        return env_response.done, env_response.observation, env_response.reward, env_response.info
    
    def generate_next_action(self):
        res = completion(
            messages=self.messages,
            tools=self.tools_info,
        )
        return res.choices[0].message
    
    def generate_precondition(self, action):
        print(type(action))
        '''
        Given the next action, generate an executable precodition that must precede the action execution.
        '''
        SYSTEM_PROMPT = f'''
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
So, remember, you cannot use var{AssertionsAgent.counter} in your precondition code.
Do not hallucinate the values of the variables.
'''
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(colored(self.tool_call_codes, 'yellow'))
        user_message = ''
        counter = 0
        for message in self.messages:
            if message["role"] == "user":
                user_message += f"customer: {message['content']}\n"
            elif message["role"] == "assistant":
                user_message += f"support: {message['content']}\n"
            elif message["role"] == "tool":
                user_message += f"python code for action: \n```python \n{self.tool_call_codes[counter]}\n"
                user_message += f"# {self.variable_defs[counter]}\n```\n"
                # user_message += f"# var_{counter} = {message['content']}\n```\n"
                counter += 1
        user_message += f"python code for the new action: \n```python\nvar_{counter} = {action.name}({(str(action.kwargs))[1:-1]})\n```\n"
        messages.append({"role": "user", "content": str(self.tools_info) + '\n' + user_message})
        print(colored(f'Context for precondition generation', 'cyan'))
        print(colored(user_message, 'cyan'))
        # return None
        res = completion(
            messages=messages,
            tools=self.tools_info,
        )
        return res.choices[0].message
    
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
        print(colored(assignments, 'magenta'))
        local_vars = {}
        for line in assignments:
            if '=' not in line:
                raise ValueError(f"Invalid assignment: {line}")
            key, value_str = map(str.strip, line.split('=', 1))

            # Heuristic: If RHS is unquoted identifier (like yusuf_rossi_9620), wrap in quotes
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value_str):
                value_str = f'"{value_str}"'

            try:
                value = ast.literal_eval(value_str)
            except Exception:
                # If ast.literal_eval fails, try JSON
                try:
                    value = json.loads(value_str)
                except Exception as e:
                    raise ValueError(f"Could not parse the value for '{key}': {e}")

            local_vars[key] = value

        return local_vars
    
    def parse_context(self, assignments):
        local_vars = {}
        for line in assignments:
            exec(line, {}, local_vars)
        return local_vars


    def run_preconditions_full(self, code: str, context: dict = None):
        errors = []

        # Step 1: parse assignment strings into actual variables
        context = self.smart_parse_context(context)

        # Step 2: prepare environment
        exec_env = {"__builtins__": __builtins__}
        exec_env.update(context)

        # Step 3: execute precondition code and catch errors
        try:
            exec(code, exec_env)
        except Exception as e:
            errors.append(f"{e.__class__.__name__}: {e}")

        return errors

    def execute_precondition(self, precondition: str):
        res = self.run_preconditions_full(precondition, self.variable_defs)
        print(res)
        return res
        errors = []

        # Define a safe execution environment
        exec_env = {
            "custom_assert": lambda cond, msg: errors.append(msg) if not cond else None,
            "__builtins__": __builtins__,  # optionally restrict built-ins
        }
        exec_env.update(self.variable_defs)

        # Replace 'assert' with 'custom_assert' in the code string
        # WARNING: naive replacement, but works if 'assert' is used in statements only
        safe_code = precondition.replace("assert ", "custom_assert(").replace(",", ", ").replace("\n", ")\n")

        try:
            exec(safe_code, exec_env)
        except Exception as e:
            errors.append(f"Runtime error: {e}")
        print(errors)
        return errors
    
    # def execute_precondition(self, precondition):
    #     local_vars = {}
    #     variable_defs = '\n'.join(self.variable_defs)
    #     exec(variable_defs, {}, local_vars)
    #     res =  exec(precondition, {}, local_vars)
    #     print(res)
    #     return res
    
    def solve(
        self, env: Env, task_index: Optional[int] = None, max_actions: int = 30, max_refinements: int = 1
    ) -> SolveResult:
        M = ''
        print('here')

        total_cost = 0.0
        reward = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        self.messages.append({"role": "user", "content": obs})
        info = env_reset_res.info.model_dump()

        print(colored(f'User: {obs}', 'blue'))
        
        solved = False
        iteration_index = 0
        while not solved and iteration_index < max_actions:
            print('Solving iteration', iteration_index)
            iteration_index += 1
            precondition_passed = False
            refinement_index = 0
            while not precondition_passed and refinement_index < max_refinements:
                refinement_index += 1
                action_message = self.generate_next_action()
                print(colored(f'Agent: {action_message}', 'green'))
                action = message_to_action(action_message)
                if action.name == RESPOND_ACTION_NAME:
                    break
                precondition = self.parse_precondition_message(self.generate_precondition(action))
                print(colored(f'Precondition: \n{precondition}', 'red'))
                precondition_res = self.execute_precondition(precondition)
                print(colored(f'Precondition result: {precondition_res}', 'magenta'))
                # precondition_passed = True
                # precondition_passed = self.execute_precondition(env, precondition)
            self.register_action(action)
            solved, response, reward, info = self.execute_action(env, action)
            print(colored(f'User: {response}', 'blue'))
            self.add_action_and_response(action_message, action, response)
            # postcondition_observation = self.check_postcondition()

        

    def solve_old(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        print(colored(f'Initial info is {info}', 'red'))
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        print('Initial task')
        print(obs)
        print()
        max_num_steps = 3
        for k in range(max_num_steps):
            print(k)
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            temp = res.choices[0].message
            print(colored(f'Agent response is {temp}', 'red'))
            next_message = model_dump(temp)
            # next_message = res.choices[0].message
            # total_cost += res._hidden_params["response_cost"]
            total_cost += res.usage.total_tokens
            action = message_to_action(next_message)
            print(colored(f'Action is {action}', 'magenta'))
            env_response = env.step(action)
            print(colored(env_response, 'green'))
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            # print(colored(f'Next set of messages is {messages}', 'cyan'))
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})

