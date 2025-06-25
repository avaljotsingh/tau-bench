import json
from typing import List, Optional, Dict, Any

from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.utils import *
from tau_bench.globals import *
from tau_bench.types import Action, RESPOND_ACTION_NAME
from tau_bench.envs.base import Env
from tau_bench.agents.state import *

class ActionAgentWithTask():
    def __init__(self, tools_info, wiki, env):
        self.tools_info = tools_info
        self.wiki = wiki
        self.env = env

    def generate_next_action_message(self, messages):
        res = completion(messages=messages, tools=self.tools_info)
        return res.choices[0].message
    
    def get_context(self, records, information, task):
        vars_from_records = ""
        for i in range(len(records.plan)):
            vars_from_records += f"{records.plan[i]}\n"
            if i < len(records.plan_outputs):
                vars_from_records += f"# {records.plan_outputs[i]}\n"
        context = [{"role": "system", "content": self.wiki}]
        if len(list(information.keys())) != 0:
            message = f"From the conversation with the customer, I have collected the following information: {information}.\n"
        else:
            message = ""
        if vars_from_records != "":
            message += f"From the conversation with the customer, I have executed the following tasks: {vars_from_records}.\n"
        else:
            message += ""
            
        message += f'''
I have collected several tasks from the conversation with the user.
Out of those, the concrete task that we need to solve now is: {task.get_description()}.
In order to solve the task, you may need to fill in the arguments in the task from the current information collected so far.
Or you may either need to choose a function call, or you may need to create a new task.
Or you may need to interact with the user to get some missing information or something else.
So, you can do 4 things.
1. Create a new task that needs to be done before the current one.
Here is a possible list of possible tasks types. validate_user,
find_user_by_email,
find_user_by_zip,
get_user_details,
get_product_details,
get_order_details,
get_user_input,
list_all_products,
calculate,
think,
transfer_to_human_agent,
cancel_pending_order,
modify_pending_order_address,
modify_pending_order_items,
modify_pending_order_payment,
modify_user_address,
return_delivered_order_items,
exchange_delivered_order_items.
For the create task, give the output in the following format:
###CREATE_TASK###
{{
  "tasktype": ,
  "args": 
}}
2. If you have anough data to execute the function call mentioned in the task description, you should execute it. 
You can return the exact python code to execute the corresponding function call.
You can just return the task type and the corresponding function call and the arguments.
Here is a possible list of functions to call:
find_user_by_email,
find_user_by_zip,
get_user_details,
get_product_details,
get_order_details,
get_user_input,
list_all_products,
calculate,
think,
transfer_to_human_agent,
cancel_pending_order,
modify_pending_order_address,
modify_pending_order_items,
modify_pending_order_payment,
modify_user_address,
return_delivered_order_items,
exchange_delivered_order_items.
For the execute function, give the output in the following format:
###EXECUTE_TASK###
{{
  function_call(arg1= arg1, arg2= arg2, ...) 
}}
Do not hallucinate and create your own functions. 
3. If u want to interact with the user to get some information, responsond in the following format:
###INTERACT_WITH_USER###
{{
  "content": 
}}
4. Finally, if you want to fill in the missing information, respond in the following format:
###FILL_IN_MISSING_INFO###
{{
  "content":
}}
'''
        context.append({"role": "user", "content": message})
        return context
    
    
    def generate_next_action(self, records, information, task):
        context = self.get_context(records, information, task)
        action_message = self.generate_next_action_message(context)
        action_type, action = self.interpret_action_message(action_message)
        return action_type, action
    
    
    def interpret_action_message(self, action_message):
        if action_message['content'] is None:
            return 'take_action', self.message_to_action(action_message) 
        elif "###EXECUTE_TASK###" in action_message['content']:
            return 'take_action', self.extract_action(action_message['content'])
        elif "###CREATE_TASK###" in action_message['content']:
            return 'create_task', self.extract_task(action_message['content'])
        elif "###INTERACT_WITH_USER###" in action_message['content']:
            return 'interact_with_user', self.extract_interaction_message(action_message['content'])
        elif "###FILL_IN_MISSING_INFO###" in action_message['content']:
            return 'fill_in_missing_info', None
        else:
            print(action_message['content'])
            raise f'Action message: {action_message['content']} not recognized'
    
    
    
    def message_to_action(self, action_message: Dict[str, Any]) -> Action:
        if "tool_calls" in action_message and action_message["tool_calls"] is not None and len(action_message["tool_calls"]) > 0 and action_message["tool_calls"][0]["function"] is not None:
            tool_call = action_message["tool_calls"][0]
            return Action(
                name=tool_call["function"]["name"],
                kwargs=json.loads(tool_call["function"]["arguments"]),
            )
        else:
            return Action(name=RESPOND_ACTION_NAME, kwargs={"content": action_message["content"]})
    
    def extract_action(self, action_message):
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
    

    def extract_task(self, action_message):
        _, json_part = action_message.split("###CREATE_TASK###")
        task_data = json.loads(json_part.strip())
        task_type_str = task_data["tasktype"]
        arguments = task_data.get("args", {})
        new_task = Task(TaskType(task_type_str), args=arguments)
        return new_task
    

    def extract_interaction_message(self, action_message):
        _, json_part = action_message.split("###INTERACT_WITH_USER###", 1)
        content = json.loads(json_part.strip())["content"]
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
        return action






















class TaskCreator:
    def __init__(self):
        pass

    def create_task(self, action_message):
        SYSTEM_PROMPT = f'''
You are a helpful formal assistant. The user is talking to a customer support agent.
Fromn the last user message, we realised that the user wants us to do something that needs a new task creation.
Can you interpret the message and create a new task?
Here is a possible list of possible tasks types. validate_user,
find_user_by_email,
find_user_by_zip,
get_user_details,
get_product_details,
get_order_details,
get_user_input,
list_all_products,
calculate,
think,
transfer_to_human_agent,
cancel_pending_order,
modify_pending_order_address,
modify_pending_order_items,
modify_pending_order_payment,
modify_user_address,
return_delivered_order_items,
exchange_delivered_order_items.
For the create task, give the output in the following format:
###CREATE_TASK###
{{
  "tasktype": ,
  "args": 
}}
        '''
        context = [{"role": "system", "content": SYSTEM_PROMPT}]
        context.append({"role": "user", "content": action_message})
        res = completion(messages=context)
        action_message = (res.choices[0].message['content'])
        print(action_message)
        return action_message
        