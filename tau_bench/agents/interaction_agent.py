import json
from termcolor import colored
from typing import Dict 

from tau_bench.agents.utils import Records
from tau_bench.envs.base import Env, Action, RESPOND_ACTION_NAME
from tau_bench.trapi_infer import completion


class InteractionAgent:
    def __init__(self, wiki, env: Env):
        self.wiki = wiki
        self.env = env

    def interact_with_user(self, context):
        print(colored('###########Starting Interaction###########', 'red'))
        done = False
        while(not done):
            res = completion(messages=context)
            action_message = res.choices[0].message['content']
            if '###INTERACT_WITH_USER###' in action_message:
                if 'content' in action_message:
                    _, json_part = action_message.split("###INTERACT_WITH_USER###", 1)
                    action_content = json.loads(json_part.strip())["content"]
                else:
                    action_content = action_message.split("###INTERACT_WITH_USER###")[1].strip()
            else:
                action_content = action_message
            print(colored(action_content, 'yellow'))
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": action_content})
            done, obs, reward, info = self.execute_action(action, self.env)
            print(colored(obs, 'blue'))
            context += [{"role": "assistant", "content": action_message}, {"role": "user", "content": obs}]
            end_interaction_response = self.end_interaction(context)
            if not 'continue' in end_interaction_response:
                break
        print(colored('###########Ending Interaction###########', 'red'))
        return end_interaction_response, obs

    def interact_with_user_no_task(self, records: Records, information: Dict):
        if len(list(information.keys())) != 0:
            message = f"From the conversation with the customer, I have collected the following information: {information}.\n"
        else:
            message = ""
        vars_from_records = ""
        for i in range(len(records.plan)):
            vars_from_records += f"{records.plan[i]}\n"
            if i < len(records.plan_outputs):
                vars_from_records += f"# {records.plan_outputs[i]}\n"
        message += f'We have already performed the following tasks: {vars_from_records}.\n'
        message += f'Can you further interact with the customer to find out what they want.\n'
        message += f'From the interaction, you can create a task to be performed.\n'
        message += f'Generate your response in the following format: ###INTERACT_WITH_USER###{{message to the user}}\n'
        message = self.wiki + '\n' + message
        context = [{"role": "system", "content": message}]

        return self.interact_with_user(context)
    
    def interact_with_user_with_task(self, current_task, action):
        system_message = f"You are a helpful customer support agent. We need some information from, a customer to complete a task. \
            Can you communicate with the user to get the relevant information? \
            Here is the task that needs to be completed: {current_task}." 
        system_message += f'Generate your response in the following format: ###INTERACT_WITH_USER###{{message to the user}}\n'
        done, obs, reward, info = self.execute_action(action, self.env)
        context = [{"role": "system", "content": system_message}, {"role": "assistant", "content": action.kwargs["content"]}, {"role": "user", "content": obs}]
        print(colored('###########Starting Interaction###########', 'red'))
        print(colored(action.kwargs['content'], 'yellow'))
        print(colored(obs, 'blue'))
        print(colored('###########Ending Interaction###########', 'red'))
        end_interaction_response = self.end_interaction(context)
        if 'continue' in end_interaction_response:
            return self.interact_with_user(context)
        else:
            return end_interaction_response, obs

    def end_interaction(self, messages):
        SYSTEM_PROMT = """
You are a helpful message classifier. You will be provided a conversation between an agent and a customer.
Based on the last message of the customer, you need to decide 
whether the user has provided with some information that must be stored or a new task to create 
or there is no material in the response whatsoever so the agent needs to continue the interaction.
You will respond with one of the following:
- store: if the user has provided with some information that must be stored.
- continue: if the agent needs to continue the interaction.
- create_task: if the user has provided with a new task to create.
- both: if the user has provided with some information that must be stored and a new task to create.
        """
        message = ""
        for i in range(len(messages)):
            message += messages[i]["role"] + ": " + messages[i]["content"] + "\n"
        context = [{"role": "system", "content": SYSTEM_PROMT}, {"role": "user", "content": message}]
        response = completion(messages=context).choices[0].message['content']
        print(colored(f'End message: {response}', 'red'))
        return response
    
    def execute_action(self, action: Action, env: Env):
        env_response = env.step(action)
        done, observation, reward, info = env_response.done, str(env_response.observation), env_response.reward, env_response.info
        return done, observation, reward, info
