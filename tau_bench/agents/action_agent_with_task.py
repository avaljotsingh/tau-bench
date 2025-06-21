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
    
    def get_context(self, information, task):
        # vars_from_records = ""
        # for i in range(len(records.plan)):
        #     vars_from_records += f"{records.plan[i]}\n"
        #     if i < len(records.plan_outputs):
        #         vars_from_records += f"# {records.plan_outputs[i]}\n"
        context = [{"role": "system", "content": self.wiki}]
        # if vars_from_records != "":
        #     message = f"From the conversation with the customer, I have collected the following information: {vars_from_records}.\n"
        # else:
        #     message = ""
        if len(list(information.keys())) != 0:
            message = f"From the conversation with the customer, I have collected the following information: {information}.\n"
        else:
            message = ""
            
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
    
    def generate_next_action(self, records, task):
        start_time = time.time()
        context = self.get_context(records, task)
        action_message = self.generate_next_action_message(context)
        print(colored(action_message, 'yellow'))
        return action_message