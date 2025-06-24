import json 
from typing import List, Optional, Dict, Any

from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.utils import *
from tau_bench.globals import *
from tau_bench.types import Action, RESPOND_ACTION_NAME
from tau_bench.envs.base import Env
from tau_bench.agents.state import *
from tau_bench.agents.action_agent_with_task import ActionAgentWithTask 
from tau_bench.agents.action_agent_with_task import ActionAgentWithoutTask 
from tau_bench.agents.action_agent_with_task import TaskCreator
from tau_bench.agents.precondition_agent import PreconditionAgent
from tau_bench.agents.postcondition_agent import PostconditionAgent
from tau_bench.agents.information_manager import InformationManager
from tau_bench.agents.task_manager import TaskManager



class Orchestrator:
    def __init__(self, tools_info: List[Dict[str, Any]], wiki: str, model: str, provider: str, temperature: float = 0.0):
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

    def interpret_action_message(self, action_message):
        if "###CREATE_TASK###" in action_message:
            return 'create_task'
        elif "###EXECUTE_TASK###" in action_message:
            return 'take_action'
        elif "###INTERACT_WITH_USER###" in action_message:
            return 'interact_with_user'
        else:
            print(action_message)
            raise f'Action message: {action_message} not recognized'
        
    def create_task(self, current_task, action_message):
        def extract_task(action_message):
            _, json_part = action_message.split("###CREATE_TASK###")
            task_data = json.loads(json_part.strip())
            task_type_str = task_data["tasktype"]
            arguments = task_data.get("args", {})
            new_task = Task(TaskType(task_type_str), args=arguments)
            return new_task
        new_task = extract_task(action_message)
        self.task_manager.add_task(new_task)
        if current_task is not None:
            self.task_manager.add_dependency(current_task, new_task)
    
    def take_action(self, action_message: str, extracted=False) -> Action:
        def extract_action(action_message):
            # Step 1: Extract the portion after the marker
            json_part = action_message.strip().split("###EXECUTE_TASK###")[-1].strip()

            # Try loading it as JSON
            try:
                data = json.loads(json_part)
                raw_call = data.pop("function_call", None)
                if raw_call is None:
                    raise ValueError("Missing 'function_call' in JSON")
                
                # Case: raw_call is just a function name
                if isinstance(raw_call, str) and '(' not in raw_call:
                    func_name = raw_call
                    if func_name.startswith("functions."):
                        func_name = func_name[len("functions."):]
                    return Action(name=func_name, kwargs=data)

                # Case: raw_call is a full call in string form
                if isinstance(raw_call, str):
                    raw_call_str = raw_call
                else:
                    raise ValueError("Unsupported format for 'function_call'")
            
            except json.JSONDecodeError:
                # Fallback: Try to extract raw function call manually
                match = re.search(r'"function_call"\s*:\s*(.+)', json_part)
                if not match:
                    raise ValueError("Invalid format: couldn't extract function_call")

                raw_call_str = match.group(1).strip().rstrip(',').rstrip('}')
                # Handle non-string values like: functions.foo(...)
                if raw_call_str.startswith("functions."):
                    raw_call_str = raw_call_str

            # Step 2: Parse function name and arguments
            func_match = re.match(r'(?:functions\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)', raw_call_str)
            if not func_match:
                raise ValueError(f"Could not parse function call: {raw_call_str}")

            func_name = func_match.group(1)
            args_str = func_match.group(2)

            kwargs = {}
            if args_str.strip():
                try:
                    fake_call = f"f({args_str})"
                    expr = ast.parse(fake_call, mode='eval')
                    call = expr.body
                    if not isinstance(call, ast.Call):
                        raise ValueError("Parsed expression is not a function call")

                    for kw in call.keywords:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)

                except Exception as e:
                    raise ValueError(f"Failed to parse arguments: {args_str}") from e

            return Action(name=func_name, kwargs=kwargs)
        if not extracted:
            action = extract_action(action_message)
        else:
            action = action_message
        print(colored('###########Taking Action###########', 'red'))
        print(colored(action, 'yellow'))
        self.register_message(Role.TOOL_CALL, action)
        self.done, obs, self.reward, info = self.execute_action(action, self.env)
        self.register_message(Role.TOOL_OUTPUT, obs)
        print(colored(obs, 'blue'))
        print(colored('###########Finished Action###########', 'red'))
    
    def interact_with_user(self, current_task, action_message):
        print(colored('###########Starting Interaction###########', 'red'))
        def extract_action(action_message):
            _, json_part = action_message.split("###INTERACT_WITH_USER###", 1)
            content = json.loads(json_part.strip())["content"]
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
            return action
        action = extract_action(action_message)
        print(colored(action.kwargs["content"], 'yellow'))
        self.done, user_response, self.reward, info = self.execute_action(action, self.env)
        print(colored(user_response, 'blue'))
        messages = [{"role": "assistant", "content": action.kwargs["content"]}, {"role": "user", "content": user_response}]
        SYSTEM_PROMPT = f"""
You are a helpful customer support agent. We need some information from, a customer to complete a task.
Can you communicate with the user to get the relevant information?
Here is the task that needs to be completed: {current_task}.
        """
        fixed_context = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            end_interaction = self.end_interaction(messages)
            if not 'continue' in end_interaction:
                break
            context = fixed_context + messages
            agent_response = completion(messages=context).choices[0].message['content']
            agent_response += 'Please give only the requested information.'
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": agent_response})
            self.done, user_response, self.reward, info = self.execute_action(action, self.env)
            messages.append({"role": "assistant", "content": agent_response})
            messages.append({"role": "user", "content": user_response})
            print(colored(agent_response, 'yellow'))
            print(colored(user_response, 'blue'))
        print(end_interaction)
        print(colored('###########Interation Ended###########', 'red'))
        if 'store' in end_interaction:
            self.information_manager.update_information(user_response)
        elif 'create_task' in end_interaction:
            action_message = self.task_creator.create_task(user_response)
            self.create_task(None, action_message)
        elif 'both' in end_interaction:
            self.information_manager.update_information(user_response)
            action_message = self.task_creator.create_task(user_response)
            self.create_task(None, action_message)

    def end_interaction(self, messages):
        SYSTEM_PROMT = """
You are a helpful message classifier. You will be provided a conversation between an agent and a customer.
Based on the last message of the customer, you need to decide 
whether the user has provided with some information that must be stored or a new task to create 
or there is no material in the response whatsoever so the agent needs to continue the interaction.
You will respond with one of the following:
- "store": if the user has provided with some information that must be stored.
- "continue": if the agent needs to continue the interaction.
- "create_task": if the user has provided with a new task to create.
- "both": if the user has provided with some information that must be stored and a new task to create.
        """
        message = ""
        for i in range(len(messages)):
            message += messages[i]["role"] + ": " + messages[i]["content"] + "\n"
        context = [{"role": "system", "content": SYSTEM_PROMT}, {"role": "user", "content": message}]
        response = completion(messages=context).choices[0].message['content']
        print(colored('Checking message end', 'red'))
        print(colored(response, 'yellow'))
        return response


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
        done, observation, reward, info = env_response.done, str(env_response.observation), env_response.reward, env_response.info
        return done, observation, reward, info

    def process_action(self, current_task, action_message):
        print(current_task, action_message)
        if action_message['content'] is None:
            action_message = self.message_to_action(action_message)
            action_type = 'take_action'
            action_extracted = True
            # self.execute_action(action, self.env)
            # return
        else:
            action_extracted = False
            action_message = action_message['content']
            action_type = self.interpret_action_message(action_message)
        print(colored('The action type is: ' + action_type + ' extracted action is' + str(action_extracted) + ' action is: ' + str(action_message), 'red'))
        if action_type == 'create_task':
            self.create_task(current_task, action_message)
        elif action_type == 'take_action':
            self.take_action(action_message, action_extracted)
            self.task_manager.remove_task(current_task)
        elif action_type == 'interact_with_user':
            self.interact_with_user(current_task, action_message)
        else:
            print(action_type)
            raise f'Action type: {action_type} not recognized'
        
    def solve(self, env: Env, task_index: Optional[int] = None, max_actions: int = 30, max_refinements: int = 2) -> SolveResult:
        self.records = Records()
        self.action_agent = ActionAgentWithTask(tools_info=self.tools_info, wiki=self.wiki, env=env)
        self.action_agent_without_task = ActionAgentWithoutTask(tools_info=self.tools_info, wiki=self.wiki, env=env)
        self.task_manager = TaskManager(TaskGraph())
        self.task_manager.add_task(Task(TaskType.ValidateUser))
        self.information_manager = InformationManager({})
        self.task_creator = TaskCreator()

        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        self.register_message(Role.USER, obs)
        self.env = env

        self.done = False
        self.reward = 0.0
        action_iteration = 0
        while not self.done and action_iteration < max_actions:
            print(action_iteration)
            action_iteration += 1
            task = self.task_manager.get_next_task()
            if task is None:
                action_message = self.action_agent_without_task.generate_next_action(self.records, self.information_manager.information)
            else:
                print(task.get_description())
                action_message = self.action_agent.generate_next_action(self.records, self.information_manager.information, task)
            self.process_action(task, action_message)

        return SolveResult(
            done=self.done,
            reward=self.reward,
            info=info,
        )