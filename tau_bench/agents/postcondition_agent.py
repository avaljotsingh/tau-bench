from tau_bench.trapi_infer import completion, model_dump
from tau_bench.agents.utils import *
from tau_bench.globals import *

POSTCONDITION_SYSTEM_PROMPT = f'''
Hi. You are an expert at planning tasks.
Here is a list of tasks that represent the intentions of the customer. 
However, the in the recent conversation with a support agent, you may need to update the tasks to better reflect the customer's needs.
Following is the existing set of tasks, followed by the recent conversation between the customer and the support.
I have represented the actions taken by the support agent in python code.
Output only the new postconditions that you discover fronm the conversation.
Output in a numbered list in different lines.
Do no repeat any tasks that is already known.
'''
        

POSTCONDITION_CHECK_SYSTEM_PROMPT = f'''
Hi. You are an expert at planning tasks.
Here is the list of actions taken byu the support agent. 
Following is one of the tasks that represent the intentions of the customer. 
Do you think that the task is solved by the actions taken by the support agent.
Output only true of false.
'''

class PostconditionAgent():
    def __init__(self, tools_info, postconditions=[]):
        self.tools_info = tools_info
        self.conversation_index = 0

    def parse_postcondition(self, postcondition_message):
        matching_lines = []
        for line in postcondition_message.splitlines():
            match = re.match(r'^\d+[\.\)\-\s]*', line)
            if match:
                cleaned_line = line[match.end():].strip()  
                matching_lines.append(cleaned_line)
        return matching_lines
    
    def get_message_from_records(self, start, records):
        message = ''
        for i in range(start, len(records.conversation)):
            role = records.conversation[i]['role']
            index = records.conversation[i]['index']
            if role == Role.USER:
                message += f"customer: {records.user_messages[index]}\n"
            if role == Role.ENV:
                message += f"# {records.env_messages[index]}\n```\n"
            elif role == Role.ACTION_AGENT:
                message += f"Support: {records.action_agent_messages[index]}\n"
            elif role == Role.TOOL_CALL:
                message += f"python code for action: \n```python \n{records.plan[index]}\n"
            elif role == Role.TOOL_OUTPUT:
                message += f"# {records.plan_outputs[index]}\n```\n"
        return message
    
    def get_context_for_generation(self, current_postconditions, message):
        context = [{"role": "system", "content": POSTCONDITION_SYSTEM_PROMPT}]
        message = f'tasks: {current_postconditions}\n' + message
        context.append({"role": "user", "content": message})
        return context
    
    def generate_new_postcondition(self, current_postconditions, message):
        context = self.get_context_for_generation(current_postconditions, message)
        res = completion(messages=context, tools=self.tools_info)
        new_post_condition_message = res.choices[0].message['content']
        if new_post_condition_message is None or (isinstance(new_post_condition_message, list) and len(new_post_condition_message) == 0):
            new_post_conditions = []
        else:
            new_post_conditions = self.parse_postcondition(new_post_condition_message)
        return new_post_conditions
    
    def check_postcondition(self, current_postconditions, message):
        tasks_solved = []
        task_solved_messages = []
        for task in current_postconditions:
            task_user_message = message + f"\n task: {task}\n"
            messages = [{"role": "system", "content": POSTCONDITION_CHECK_SYSTEM_PROMPT}, {"role": "user", "content": task_user_message}]
            res = completion(messages=messages)
            check_passed = res.choices[0].message['content']
            tasks_solved.append('true' in check_passed or 'True' in check_passed)
            task_solved_messages.append(check_passed)
        print(task_solved_messages)
        return tasks_solved
    
    def register_conversation(self, current_postconditions, records):
        start_time = time.time()
        message1 = self.get_message_from_records(self.conversation_index, records)
        new_postconditions = self.generate_new_postcondition(current_postconditions, message1)
        message2 = self.get_message_from_records(0, records)
        tasks_solved = self.check_postcondition(current_postconditions, message2)
        self.conversation_index = len(records.conversation)
        pending_postconditions = []
        for i in range(len(tasks_solved)):
            if not tasks_solved[i]:
                pending_postconditions.append(current_postconditions[i])
        postcondition_agent_time.record_time(time.time() - start_time)
        return pending_postconditions, new_postconditions
    