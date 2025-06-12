from typing import Any, Callable, Dict, List, Type, Optional, Set, Union, Tuple
from termcolor import colored
import traceback
import sys

from tau_bench.envs.retail.data import load_data
from tau_bench.envs.retail.tools import ALL_TOOLS

from code import Code

class PlanExecutor:
    def __init__(self, plan: Code, data_load_func=load_data, tools=ALL_TOOLS):
        self.plan = plan
        self.plan.remove_imports()
        self.data_load_func = data_load_func
        self.data = data_load_func()
        
        self.tools_map: Dict[str, Type] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }

    def reset(self):
        self.data = self.data_load_func()

    def exec_func(self, name: str, kwargs: dict):
        try:
            return self.tools_map[name].invoke(data=self.data, **kwargs)
        except Exception as e:
            return f"Error: {e}"

    def execute(self):
        tree = self.plan.code_ast

        exec_globals = {}
        exec_locals = {}

        # Correctly bind each tool_class using a function factory
        for name, tool_class in self.tools_map.items():
            def make_tool_func(tool_cls):
                return lambda **kwargs: tool_cls.invoke(data=self.data, **kwargs)
            exec_globals[name] = make_tool_func(tool_class)

        code = compile(tree, "<exec>", "exec")
        try:
            code = compile(tree, "<exec>", "exec")
            exec(code, exec_globals, exec_locals)
            print(colored("Executed successfully", "green"))

        except Exception as e:
            print(colored("Execution failed with an error:", "red"))

            # Extract traceback details
            tb = e.__traceback__
            extracted_tb = traceback.extract_tb(tb)

            # Find the frame for <exec> (this is the dynamic code)
            for frame in extracted_tb:
                if frame.filename == "<exec>":
                    lineno = frame.lineno
                    print(colored(f"\nError occurred at line {lineno} in your original code:", "red"))

                    # Print the offending line from source, if available
                    if hasattr(self.plan, "code_str") and self.plan.code_str:
                        lines = self.plan.code_str.splitlines()
                        if 0 < lineno <= len(lines):
                            print(colored(f">>> {lines[lineno - 1].strip()}", "yellow"))
                        else:
                            print(colored("⚠ Line number out of range in source code.", "yellow"))
                    else:
                        print(colored("⚠ Original source code not available to show the error line.", "yellow"))

                    break

            # Print full Python traceback
            print(colored("\nFull traceback:", "red"))
            traceback.print_exception(type(e), e, tb, file=sys.stdout)

            return {"error": str(e), "traceback": traceback.format_exception(type(e), e, tb)}

        return exec_locals


    def get_data(self):
        return self.data
