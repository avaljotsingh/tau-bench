import time 
import tabulate

class DebugFlag():
    def __init__(self):
        self.debug = False

    def set(self):
        self.debug = True

    def unset(self):
        self.debug = False

    def is_set(self):
        return self.debug
    

class ContextLength():
    def __init__(self):
        self.length = []

    def add(self, length):
        self.length.append(length)

    def get_lengths_from_messages(self, messages):
        lengths = []
        for message in messages:
            lengths.append(len(message['content']))
        self.add(lengths)


class Time():
    def __init__(self):
        self.total_time = 0

    def record_time(self, t):
        self.total_time += t

    def get_time(self):
        return self.total_time
    
def print_times():
    times = [("LLM", llm_time.get_time()), ("Env Step", env_time.get_time()), ("Action Agent", action_agent_time.get_time()), ("Precondition Agent", precondition_agent_time.get_time()), ("Postcondition Agent", postcondition_agent_time.get_time())]
    print(tabulate.tabulate(times, headers=["Source", "Time"]))
    print('Context lengths', contextLength.length)


llm_time = Time()
env_time = Time()
action_agent_time = Time()
precondition_agent_time = Time()
postcondition_agent_time = Time()
contextLength = ContextLength()