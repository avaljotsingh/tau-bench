import json
from typing import List, Optional, Dict, Any

from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.utils import *
from tau_bench.globals import *
from tau_bench.types import Action, RESPOND_ACTION_NAME
from tau_bench.envs.base import Env

class ActionAgent():
    def __init__(self, tools_info, wiki, env):
        self.tools_info = tools_info
        self.wiki = wiki
        self.env = env

    def generate_next_action_message(self, messages):
        res = completion(messages=messages, tools=self.tools_info)
        return res.choices[0].message
    
    def get_context(self, records: Records, precondition_results, current_postconditions):
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
        context.append({'role': 'user', 'content': f'Based on our conversation, I have summarised the following pending tasks: {current_postconditions}'})
        if len(precondition_results) > 0:
            context.append({'role': 'user', 'content': f'Based on our conversation, I also have the following preconditions that failed: {precondition_results}'})
        return context
    
    def generate_next_action(self, records, precondition_results, current_postconditions):
        start_time = time.time()
        context = self.get_context(records, precondition_results, current_postconditions)
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
        return env_response.done, str(env_response.observation), env_response.reward, env_response.info