from tau_bench.trapi_infer import completion

class EnvResponseAgent:
    def __init__(self):
        pass

    def check_response(self, task, action, response):
        system_message = f"You are a helpful assistant. We need some information from, a customer to complete a task. \
            I have to solve the tast {task}. I took the action {action}. I got the response {response}. \
            I don't understand if the task has been solved or not, and if I should save this information in my records. \
            Can you interpret the response and tell me? \
            Respond with one of the following: \
            - task_solved: if the task has been solved and nothing is to be stored. \
            - task_not_solved: if the task has not been solved and it is an error. \
            - save_information: if the task is solved and I should save this information in my records. "
        return completion(messages=[{"role": "system", "content": system_message}, {"role": "user", "content": response}]).choices[0].message['content']