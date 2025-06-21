from enum import Enum
from termcolor import colored
import ast
import re

class Role(str, Enum):
    USER = "USER"
    ENV = "ENV"
    ACTION_AGENT = "ACTION_AGENT"
    TOOL_CALL = "TOOL_CALL"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    PRECONDITION_AGENT = "PRECONDITION_AGENT"
    PRECONDITION_OUTPUT = "PRECONDITION_OUTPUT"
    POSTCONDITION_AGENT = "POSTCONDITION_AGENT"
    POSTCONDITION_OUTPUT = "POSTCONDITION_OUTPUT"
    

def print_message(role, message):
    if role==Role.USER:
        print(colored(f'User: \n{message}', 'blue'))
    elif role==Role.ENV:
        print(colored(f'Environment: \n{message}', 'cyan'))
    elif role==Role.ACTION_AGENT:
        print(colored(f'Action Agent: \n{message.kwargs["content"]}', 'green'))
    elif role==Role.TOOL_CALL:
        print(colored(f'Tool: \n{message}', 'yellow'))
    elif role==Role.TOOL_OUTPUT:
        print(colored(f'Tool: \n{message}', 'yellow'))
    elif role==Role.PRECONDITION_AGENT:
        print(colored(f'Precondition Agent: \n{message}', 'red'))
    elif role==Role.PRECONDITION_OUTPUT:
        print(colored(f'Precondition Output: \n{message}', 'red'))
    elif role==Role.POSTCONDITION_AGENT:
        print(colored(f'Postcondition Agent: \n{message}', 'magenta'))
    elif role==Role.POSTCONDITION_OUTPUT:
        print(colored(f'Postcondition Output: \n{message}', 'magenta'))
    else:
        print(role)
        raise f'Role: {role} not defined'


def fix_quoted_lists(code_str):
    """
    Fixes broken single-quoted strings that look like lists, e.g.,
    item_ids='['123']' → item_ids="['123']"
    """
    # Replace ='<[...']' with ="[..."]"
    return re.sub(r"=\s*'(\[.*?\])'", r'="\1"', code_str)

def extract_function_call_components(code_str):
    print(code_str)
    if 'think' in code_str:
        return 'think', {'thought': code_str.split('=')[2][:-1]}
    """
    Extract the function name and keyword arguments from an assignment
    like 'var_1 = func(a=..., b=...)', even if argument values contain
    commas, quotes, or stringified lists.
    
    Returns: (func_name: str, kwargs: dict)
    Raises: ValueError on malformed input
    """
    try:
        # Clean and fix common malformed input
        code_str = code_str.strip()
        code_str = fix_quoted_lists(code_str)

        # Extract RHS (right-hand side): everything after the first '='
        if '=' not in code_str:
            raise ValueError("No '=' found in the input")

        rhs = code_str.split('=', 1)[1].strip()

        # Parse the RHS as a function call
        tree = ast.parse(rhs, mode='eval')
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Expected a function call on the right-hand side")

        func_call = tree.body
        func_name = (
            func_call.func.id if isinstance(func_call.func, ast.Name)
            else ast.unparse(func_call.func) if hasattr(ast, 'unparse')
            else '<unknown_function>'
        )

        kwargs = {}
        for kw in func_call.keywords:
            try:
                # Safely evaluate constants
                value = ast.literal_eval(kw.value)
            except Exception:
                # Fallback: try to get the source text
                value = ast.unparse(kw.value) if hasattr(ast, 'unparse') else str(kw.value)
            kwargs[kw.arg] = value

        return func_name, kwargs

    except SyntaxError as e:
        raise ValueError(f"Syntax error while parsing: {e}")
    except Exception as e:
        raise ValueError(f"Failed to extract function call components: {e}")
    

class Records():
    def __init__(self):
        self.conversation = []
        self.user_messages = []
        self.env_messages = []
        self.action_agent_messages = []
        self.plan = []
        self.plan_outputs = []
        self.preconditions = []
        self.preconditions_outputs = []
        self.postconditions = []
        self.postconditions_outputs = []
        self.var_counter = 0

    def get_records(self):
        return {
            'conversation': [self.conversation],
            'user_messages': self.user_messages,
            'env_messages': self.env_messages,
            'action_agent_messages': self.action_agent_messages,
            'plan': self.plan,
            'plan_outputs': self.plan_outputs,
            'preconditions': self.preconditions,
            'preconditions_outputs': self.preconditions_outputs,
            'postconditions': self.postconditions,
            'postconditions_outputs': self.postconditions_outputs,
        }



    def get_messages(self):
        messages = []
        func_name, kwargs = None, None
        for i in range(len(self.conversation)):
            role = self.conversation[i]['role']
            index = self.conversation[i]['index']
            if role == Role.USER:
                messages.append({"role": "user", "content": self.user_messages[index]})
            elif role == Role.ENV:
                messages.append({"role": "tool", "content": f'# {self.env_messages[index]}'})
            elif role == Role.ACTION_AGENT:
                messages.append({"role": "assistant", "content": self.action_agent_messages[index]})
            elif role == Role.TOOL_CALL:
                func_name, kwargs = extract_function_call_components(self.plan[index])
                messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": func_name, "arguments": str(kwargs)}}]})
            elif role == Role.TOOL_OUTPUT:
                messages.append({"role": "tool", "name": func_name, "content": self.plan_outputs[index].split('=')[1].strip()})
            # elif role == Role.PRECONDITION_AGENT:
            #     messages.append({"role": "precondition agent", "content": self.preconditions[index]})
            # elif role == Role.PRECONDITION_OUTPUT:
            #     messages.append({"role": "env", "content": self.preconditions_outputs[index]})
            # elif role == Role.POSTCONDITION_AGENT:
            #     messages.append({"role": "postcondition agent", "content": self.postconditions[index]})
            # elif role == Role.POSTCONDITION_OUTPUT:
            #     messages.append({"role": "env", "content": self.postconditions_outputs[index]})
        return messages
