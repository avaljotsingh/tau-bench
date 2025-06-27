# Copyright Sierra

import json
from typing import List, Optional, Dict, Any
from termcolor import colored
# from litellm import completion

from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.globals import *

PRECONDITION_SYSTEM_PROMPT = f'''
Hi. You are an expert at doing sanity checks.
Here is a conversation between a customer and a support. 
The support has now genarated the next action. 
You need to check if this is th right action that the customer support must take. 
Further, the action may be risky or have side-effects and may need the cxustomer's approval.
If you think something is wrong, can you provide some advice to the assistant?
You can structure your output as on the the following. 
1. OK
2. ADVICE <advice>: If you need to give some advice to the customer support.
Make sure that your advice is directed to the customer support. I will pass on your message as if the customer support is thinking it verbatim
'''

class PreconditionsAgent():
    def __init__(self):
        pass 

    def generate_context(self, messages, action):
        message = ''
        for m in messages:
            if m['role'] == 'user':
                message += f'CUSTOMER: {m['content']}\n'
            elif m['role'] == 'assistant' and m['content'] is not None:
                message += f'SUPPORT STAFF: {m['content']}\n'
            elif m['role'] == 'assistant' and m['content'] is None:
                message += f'TOOL CALL: {m['tool_calls'][0]['function']['name']} with arguments {m['tool_calls'][0]['function']['arguments']}\n'
            elif m['role'] == 'tool':
                message += f'TOOL RESPONSE: {m['content']}\n'
            elif m['role'] == 'system':
                message += f'INSTRUCTIONS: {m['content']}\n'
        message += f'TOOL CALL: {action.name} with arguments {action.kwargs}\n'
        context = [{"role": "system", "content": PRECONDITION_SYSTEM_PROMPT}, {"role": "user", "content": message}]
        return context

    def parse_precondition_response(self, res):
        if 'OK' in res:
            return None 
        elif 'ADVICE' in res:
            return res.split('ADVICE:')[1].strip()
        else:
            return None
        
    def create_action_from_advice(self, action, advice):
        if advice is not None:
            return Action(
                name="think",
                kwargs={"thought": f'Thinking of the next action to be {action}. Here is an external advice: {advice}'},
            )
        else:
            return None
    
    def generate_precondition(self, messages, action):
        context = self.generate_context(messages, action)
        res = completion(messages=context).choices[0].message['content']
        return self.create_action_from_advice(action, self.parse_precondition_response(res))
    

class ToolCallingAgentWithPreconditionsAndPython(Agent):
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
        self.preconditions_agent = PreconditionsAgent()

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        thought_action_flag = False
        for k in range(max_num_steps):
            start_time = time.time()
            # print(messages)
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            temp = res.choices[0].message
            next_message = model_dump(temp)
            # next_message = res.choices[0].message
            # total_cost += res._hidden_params["response_cost"]
            total_cost += res.usage.total_tokens
            action = message_to_action(next_message)
            action_agent_time.record_time(time.time() - start_time)
            critical_functions = ['cancel_pending_order', 'exchange_delivered_order_items', 'modify_pending_order_address', 'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address', 'return_delivered_order_items', 'transfer_to_human_agents']
            if action.name != RESPOND_ACTION_NAME and not thought_action_flag and action.name in critical_functions:
                thought_action = self.preconditions_agent.generate_precondition(messages, action)
                if thought_action is not None:
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                    next_message['tool_calls'][0]['function']['name'] = "think"
                    next_message['tool_calls'][0]['function']['arguments'] = str(thought_action.kwargs)
                    messages.extend(
                        [
                            next_message,
                            {
                                "role": "tool",
                                "tool_call_id": next_message["tool_calls"][0]["id"],
                                "name": "think",
                                "content": ""
                            },
                        ]
                    )
                    thought_action_flag = True
                    continue
            if thought_action_flag and action.name != RESPOND_ACTION_NAME and action.name not in ['think']:
                thought_action_flag = False
            env_response = env.step(action)
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
            if action.name != RESPOND_ACTION_NAME:
                thought_action_2 = Action(name="think", kwargs={"thought": f'If I need to do some analysis over the data, I can write a python code to do it for reliable results.'})
                
                next_message['tool_calls'][0]['function']['name'] = "think"
                next_message['tool_calls'][0]['function']['arguments'] = str(thought_action_2.kwargs)
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": "think",
                            "content": ""
                        },
                    ]
                )
                thought_action_flag_2 = True

            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
            records={}
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
