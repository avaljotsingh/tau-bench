import json 
from typing import List, Optional, Dict, Any

from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.utils import *
from tau_bench.globals import *
from tau_bench.types import Action, RESPOND_ACTION_NAME
from tau_bench.envs.base import Env
from tau_bench.agents.state import *
from tau_bench.agents.action_agent_with_task import ActionAgentWithTask 
from tau_bench.agents.precondition_agent import PreconditionAgent
from tau_bench.agents.postcondition_agent import PostconditionAgent


class InformationManager:
    def __init__(self, information):
        self.information = information

    def clean_code_block(self, code_str: str) -> str:
        lines = code_str.strip().splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)

    def add_new_knowledge(self, new_knowledge, allowed_classes):
        """
        Executes a knowledge string and updates the `information` dictionary
        with any created variables that are instances of the allowed classes.
        
        Parameters:
            information: Dictionary to update.
            knowledge_str: A string of code that defines new knowledge (e.g., user_1 = User(...)).
            allowed_classes: A dictionary of allowed class names to class types (e.g., {"User": User, ...}).
        """
        # Limit scope to only allowed classes
        # exec_scope = allowed_classes.copy()
        new_knowledge = self.clean_code_block(new_knowledge)
        exec_scope = allowed_classes.copy()
        local_vars = {}

        exec(new_knowledge, exec_scope, local_vars)

        # Add only allowed objects to information
        for name, obj in local_vars.items():
            if any(isinstance(obj, cls) for cls in allowed_classes.values()):
                self.information[name] = obj

    def parse_user_message(self, message):

        if len(list(self.information.keys())) == 0:
            information = ""
        else:
            information = json.dumps(self.information)
        SYSTEM_PROMPT = f'''
I have the following information currently: {information}. 
The user will provide further information: {message}. 
Please update the information accordingly. 
The format shopuld be the following: 
information is a dictonary with the keys as objects of the following classes: 
You can use the following class objects: 
class User(BaseModel):
    first_name: str = "-"
    last_name: str = "-"
    zip: str = "-"
    user_id: str = "-"
    email: str = "-"

class Item(BaseModel):
    item_id: str = "-"
    properties: Dict[str, Any] = {{}}
    availability: str = "-"
    price: str = '-'

class Product(BaseModel):
    product_id: str = "-"
    product_name: str = "-"
    items: List[Item] = []

class Order(BaseModel):
    order_id: str = "-"
    items: List[Item] = []
    status: str = "-"

You should name the variables as user_1, order_1, etc.
Do not define the classes again. Just define the objects or update the information in the existing objects.
'''
        context = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}]
        res = completion(messages=context)
        new_information = res.choices[0].message.content
        print(new_information)
        # self.update_information(new_information)
        return new_information

    def update_information(self, user_message):
        new_knowledge = self.parse_user_message(user_message)
        self.add_new_knowledge(new_knowledge, allowed_classes={"User": User, "Item": Item, "Product": Product, "Order": Order})

    

class TaskManager:
    def __init__(self, task_graph):
        self.task_graph = task_graph

    def get_next_task(self):
        independent_tasks = self.task_graph.find_roots()
        return independent_tasks[0]
    
    def add_task(self, task: Task):
        self.task_graph.add_task(task)

    def remove_task(self, task: Task):
        independent_tasks = self.task_graph.find_roots()
        assert(task in independent_tasks)
        self.task_graph.remove_task(task)

    def add_dependency(self, task1: Task, task2: Task):
        self.task_graph.add_edge(task1, task2)

    def get_all_pending_tasks(self):
        pending_tasks = self.task_graph.nodes
        return pending_tasks
    


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
        def extract_tast(action_message):
            _, json_part = action_message.split("###CREATE_TASK###")
            task_data = json.loads(json_part.strip())
            task_type_str = task_data["tasktype"]
            arguments = task_data.get("args", {})
            new_task = Task(TaskType(task_type_str), args=arguments)
            return new_task
        new_task = extract_tast(action_message)
        self.task_manager.add_task(new_task)
        self.task_manager.add_dependency(current_task, new_task)
    
    def take_action(self, action_message: str) -> Action:
        def extract_action(action_message):
            func_line = action_message.strip().split("###EXECUTE_TASK###")[-1].strip()
            func_line = func_line.strip("{}").strip()
            node = ast.parse(func_line, mode='eval').body
            func_name = f"{node.func.value.id}.{node.func.attr}"
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            action = Action(name=func_name, kwargs=kwargs)
            return action
        action = extract_action(action_message)
        self.done, obs, self.reward, info = self.execute_action(action, self.env)
        print(obs)
    
    def interact_with_user(self, current_task, action_message):
        def extract_action(action_message):
            _, json_part = action_message.split("###INTERACT_WITH_USER###", 1)
            content = json.loads(json_part.strip())["content"]
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
            return action
        action = extract_action(action_message)
        self.done, user_response, self.reward, info = self.execute_action(action, self.env)
        messages = [{"role": "assistant", "content": action.kwargs["content"]}, {"role": "user", "content": user_response}]
        SYSTEM_PROMPT = f"""
You are a helpful customer support agent. We need some information from, a customer to complete a task.
Can you communicate with the user to get the relevant information?
Here is the task that needs to be completed: {current_task}.
        """
        fixed_context = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            if self.end_interaction(messages):
                break
            context = fixed_context + messages
            agent_response = completion(messages=context).choices[0].message['content']
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": agent_response})
            self.done, user_response, self.reward, info = self.execute_action(action, self.env)
            messages.append({"role": "assistant", "content": agent_response})
            messages.append({"role": "user", "content": user_response})
        print(colored(user_response, 'red'))
        self.information_manager.update_information(user_response)

    def end_interaction(self, messages):
        SYSTEM_PROMT = """
You are a helpful message classifier. You will be provided a conversation between an agent and a customer.
Based on the last message of the customer, you need to decide 
whether the user has provided with some information that must be stored or the agent needs to continue the interaction.
You will respond with one of the following:
- "store": if the user has provided with some information that must be stored.
- "continue": if the agent needs to continue the interaction.
        """
        message = ""
        for i in range(len(messages)):
            message += messages[i]["role"] + ": " + messages[i]["content"] + "\n"
        context = [{"role": "system", "content": SYSTEM_PROMT}, {"role": "user", "content": message}]
        response = completion(messages=context).choices[0].message['content']
        return response == "store"


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
        return env_response.done, str(env_response.observation), env_response.reward, env_response.info

    def process_action(self, current_task, action_message):
        if action_message['content'] is None:
            action = self.message_to_action(action_message)
            self.execute_action(action, self.env)
            return
        action_message = action_message['content']
        action_type = self.interpret_action_message(action_message)
        if action_type == 'create_task':
            self.create_task(current_task, action_message)
        elif action_type == 'take_action':
            self.take_action(action_message)
        elif action_type == 'interact_with_user':
            self.interact_with_user(current_task, action_message)
        else:
            print(action_type)
            raise f'Action type: {action_type} not recognized'
        
    def solve(self, env: Env, task_index: Optional[int] = None, max_actions: int = 3, max_refinements: int = 2) -> SolveResult:
        self.records = Records()
        self.action_agent = ActionAgentWithTask(tools_info=self.tools_info, wiki=self.wiki, env=env)
        self.task_manager = TaskManager(TaskGraph())
        self.task_manager.add_task(Task(TaskType.ValidateUser))
        self.information_manager = InformationManager({})

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
            print(task.get_description())
            action_message = self.action_agent.generate_next_action(self.information_manager.information, task)
            self.process_action(task, action_message)
            continue
            action_type = self.interpret_action_message(action_message)
            if action_type == 'create_task':
                new_task = self.create_task(action_message)
                print(new_task.get_description())
                self.task_manager.add_task(new_task)
                self.task_manager.add_dependency(task, new_task)
            elif action_type == 'take_action':
                action = self.take_action(action_message)
                kxdfj
                if action.name == RESPOND_ACTION_NAME:
                    action_role = Role.ACTION_AGENT 
                else:
                    action_role = Role.TOOL_CALL 
                self.done, obs, reward, info = self.execute_action(action, env)
                self.task_manager.remove_task(task)
            elif action_type == 'interact_with_user':
                self.done, obs, reward, info = self.interact_with_user(action_message)
                self.information_manager.update_information(obs)
            else:
                raise f'Action type: {action_type} not recognized'
        if self.done:
            print(self.task_manager.get_all_pending_tasks())
            kjsdfh