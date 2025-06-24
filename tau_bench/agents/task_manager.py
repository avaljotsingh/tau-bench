from tau_bench.agents.state import Task, TaskGraph

class TaskManager:
    def __init__(self, task_graph: TaskGraph):
        self.task_graph = task_graph

    def get_next_task(self):
        independent_tasks = self.task_graph.find_roots()
        if len(independent_tasks) == 0:
            return None
        return independent_tasks[0]
    
    def add_task(self, task: Task):
        self.task_graph.add_task(task)

    def remove_task(self, task: Task):
        independent_tasks = self.task_graph.find_roots()
        if task not in independent_tasks:
            print(task, independent_tasks)
        assert(task in independent_tasks)
        self.task_graph.nodes.remove(task)
        for task_ in self.task_graph.nodes:
            if task in self.task_graph.edges[task_]:
                self.task_graph.edges[task_].remove(task)

    def add_dependency(self, task1: Task, task2: Task):
        # Task 1 depends on task 2
        self.task_graph.add_edge(task1, task2)

    def get_all_pending_tasks(self):
        pending_tasks = self.task_graph.nodes
        return pending_tasks
    