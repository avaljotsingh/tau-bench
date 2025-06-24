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
2. You can return the exact python code to execute the corresponding function call.
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
        start_time = time.time()
        context = self.get_context(records, information, task)
        action_message = self.generate_next_action_message(context)
        return action_message
    

class ActionAgentWithoutTask():
    def __init__(self, tools_info, wiki, env):
        self.tools_info = tools_info
        self.wiki = wiki
        self.env = env

    def generate_next_action_message(self, messages):
        res = completion(messages=messages, tools=self.tools_info)
        return res.choices[0].message
    
    def get_context(self, records, information):
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
        message += f'We have already performed the following tasks: {vars_from_records}.\n'
        message += f'Can you further interact with the customer to find out what they want.\n'
        message += f'From the interaction, you can create a task to be performed.\n'
        message += f'''
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
If u want to interact with the user to get some information, responsond in the following format:
###INTERACT_WITH_USER###
{{
  "content": 
}}
        '''
        context.append({"role": "user", "content": message})
        return context
    
    def generate_next_action(self, records, information):
        start_time = time.time()
        context = self.get_context(records, information)
        action_message = self.generate_next_action_message(context)
        action = (action_message)
        action_agent_time.record_time(time.time() - start_time)
        return action
    
class TaskCreator:
    def __init__(self):
        pass

    def create_task(self, action_message):
        SYSTEM_PROMPT = f'''
You are a hlpful formal assistant. The user is talking to a customer support agent.
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
        