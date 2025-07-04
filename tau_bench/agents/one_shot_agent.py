# Copyright Sierra

import json
# from litellm import completion
from tau_bench.trapi_infer import completion, model_dump
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from termcolor import colored

class OneShotAgent(Agent):
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

        res = completion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            # tools=self.tools_info,
            # temperature=self.temperature,
        )

        messages.append(model_dump(res.choices[0].message))
        total_cost += res.usage.total_tokens

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

        


# def message_to_action(
#     message: Dict[str, Any],
# ) -> Action:
#     if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
#         tool_call = message["tool_calls"][0]
#         return Action(
#             name=tool_call["function"]["name"],
#             kwargs=json.loads(tool_call["function"]["arguments"]),
#         )
#     else:
#         return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
