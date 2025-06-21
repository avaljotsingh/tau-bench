from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, RESPOND_ACTION_NAME
from tau_bench.globals import *
from tau_bench.agents.utils import *
from tau_bench.agents.action_agent import ActionAgent
from tau_bench.agents.precondition_agent import PreconditionAgent
from tau_bench.agents.postcondition_agent import PostconditionAgent



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
        

    def solve(self, env: Env, task_index: Optional[int] = None, max_actions: int = 30, max_refinements: int = 2) -> SolveResult:
        self.records = Records()
        self.action_agent = ActionAgent(tools_info=self.tools_info, wiki=self.wiki, env=env)
        self.precondition_agent = PreconditionAgent(tools_info=self.tools_info)
        self.postcondition_agent = PostconditionAgent(tools_info=self.tools_info)
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        self.register_message(Role.USER, obs)
        self.postconditions = []
        

        done = False
        action_iteration = 0
        while not done and action_iteration < max_actions:
            action_iteration += 1
            precondition_passed = False
            precondition_results = []
            refinement_iteration = 0
            while not precondition_passed and refinement_iteration < max_refinements:
                refinement_iteration += 1
                print(f'Task Index: {task_index}, Action Iteration: {action_iteration}, Refinement Iteration: {refinement_iteration}')
                # print(action_iteration, refinement_iteration)
                print_times()
                action = self.action_agent.generate_next_action(self.records, precondition_results, self.postconditions)
                if action.name == RESPOND_ACTION_NAME:
                    action_role = Role.ACTION_AGENT 
                    precondition_passed = True
                else:
                    action_role = Role.TOOL_CALL 
                    precondition_passed = False
                    precondition = self.precondition_agent.generate_precondition(self.records, action)
                    self.register_message(Role.PRECONDITION_AGENT, precondition)
                    precondition_results = self.precondition_agent.execute_precondition(precondition, self.records)
                    precondition_passed = len(precondition_results) == 0
                    self.register_message(Role.PRECONDITION_OUTPUT, precondition_passed)
            self.register_message(action_role, action)
            done, observation, reward, env_info = self.action_agent.execute_action(action, env)
            # assert(reward < 1 or 'STOP' in observation)
            # assert(reward > 0 or not 'STOP' in observation)
            info = {**info, **env_info.model_dump()}
            self.register_output_from_action(action, observation)
            # old_postconditions, new_postconditions = self.postcondition_agent.register_conversation(self.postconditions, self.records)
            # self.register_message(Role.POSTCONDITION_AGENT, old_postconditions)
            # self.register_message(Role.POSTCONDITION_AGENT, new_postconditions)
            # self.postconditions = old_postconditions + new_postconditions
        
        messages = self.records.get_messages()
        return SolveResult(reward=reward, total_cost=0.0, info=info, messages=messages, records=self.records.get_records())
    