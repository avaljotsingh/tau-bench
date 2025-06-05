import ast
import json
import builtins
import tokenize
import io
from termcolor import colored

from dead_code import no_dead_code

def remove_comments(code_str):
    """Remove all comments from Python code using tokenize."""
    tokens = list(tokenize.generate_tokens(io.StringIO(code_str).readline))
    tokens_no_comments = [tok for tok in tokens if tok.type != tokenize.COMMENT]
    code_no_comments = tokenize.untokenize(tokens_no_comments)
    # untokenize may return bytes or str depending on Python version
    if isinstance(code_no_comments, bytes):
        code_no_comments = code_no_comments.decode('utf-8')
    return code_no_comments

def add_line_numbers(code_str):
    """Add line numbers to each non-empty line of code string."""
    lines = code_str.splitlines()
    numbered_lines = []
    line_num = 1
    for line in lines:
        if line.strip():  # skip empty lines
            numbered_lines.append(f"{line_num:3d}: {line}")
            line_num += 1
    return "\n".join(numbered_lines)

def pretty_print_code(code_str, line_numbers=True):
    if line_numbers:
        print(add_line_numbers(code_str))
    else:
        print(code_str)

def clean_code(code):
    lines = code.strip().split('\n')
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines[-1].startswith("```"):
        lines = lines[:-1]

    return remove_comments("\n".join(lines))


def is_correct_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def has_valid_function_calls(code_str, allowed_functions=['get_input_from_user']):
    """
    Analyze function calls in code_str.
    
    Args:
      code_str (str): Python source code as string.
      allowed_functions (set or None): Optional set of additional allowed function names.
    
    Returns:
      tuple:
        - bool: True if all function calls are allowed, False otherwise
        - set: disallowed function names (empty if none)
        - dict: mapping {function_name: source_library} for all called functions
    """
    if allowed_functions is None:
        allowed_functions = set()
    else:
        allowed_functions = set(allowed_functions)
        
    tree = ast.parse(code_str)
    
    # Track imports
    imported_modules = {}
    imported_functions = {}
    
    # Collect imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                imported_functions[alias.asname or alias.name] = module
    
    builtin_funcs = set(dir(builtins))
    
    disallowed_calls = set()
    func_to_lib = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = None
            lib_name = None
            
            if isinstance(node.func, ast.Name):
                # e.g. foo()
                func_name = node.func.id
                if func_name in imported_functions:
                    lib_name = imported_functions[func_name]
                elif func_name in builtin_funcs:
                    lib_name = 'built-in'
                elif func_name in allowed_functions:
                    lib_name = 'allowed custom'
                else:
                    lib_name = 'local or unknown'
                    # Mark disallowed if not in allowed_functions
                    if func_name not in allowed_functions:
                        disallowed_calls.add(func_name)
            
            elif isinstance(node.func, ast.Attribute):
                attr = node.func
                attrs = []
                while isinstance(attr, ast.Attribute):
                    attrs.append(attr.attr)
                    attr = attr.value
                if isinstance(attr, ast.Name):
                    attrs.append(attr.id)
                    attrs.reverse()
                    base_name = attrs[0]
                    func_name = attrs[-1]
                    
                    if base_name in imported_modules:
                        lib_name = imported_modules[base_name]
                    elif func_name in allowed_functions:
                        lib_name = 'allowed custom'
                    elif base_name in builtin_funcs:
                        lib_name = 'built-in'
                    else:
                        # This is a method call on a variable (like list.append)
                        lib_name = 'method on variable or unknown'
                        # Do NOT add to disallowed_calls
                else:
                    # Complex expression, assume method on variable; allow it
                    continue  # Skip flagging as disallowed
            
            if func_name:
                func_to_lib[func_name] = lib_name
    
    all_allowed = len(disallowed_calls) == 0
    return all_allowed, disallowed_calls, func_to_lib

def static_check(code_str):
    if not is_correct_syntax(code_str):
        return False, 'The code generated is syntactically wrong'
    
    code_ast = ast.parse(code_str)
    flag, invalid_funcs, func_to_liob_map = has_valid_function_calls(code_ast)
    if not flag:
        return False, f'The code generated has invalid function calls. The incorrect functions are {invalid_funcs}'
    
    flag, dead_lines, unused_vars = no_dead_code(code_ast)
    if not flag:
        return False, f'The code generated has dead code. The dead lines are {dead_lines} and the unused variables are {unused_vars}'

    return True, 'The code generated is correct'

def analyse_codes(filepath, import_statements):
    data = json.load(open(filepath))
    for i, task in enumerate(data):
        code_str = clean_code(task['traj'][2]['content'])
        code_str = ''.join(import_statements) + code_str

        flag, comment = static_check(code_str)
        if not flag:
            print(f'Task {i}: ', colored(comment, 'red'))
        else:
            print(f'Task {i}: ', colored(comment, 'green'))
    
    print_code(data[1])



def print_code(task):
    code_str = clean_code(task['traj'][2]['content'])
    code_str = ''.join(import_statements) + code_str

    pretty_print_code(code_str, False)


file_path = 'one-shot-gpt-4o-0.0_range_0--1_user-gpt-4o-one-shot_0603172050.json'
allowed_functions = ['calculate', 'cancel_pending_order', 'exchange_delivered_order_items', 'find_user_id_by_email', 'find_user_id_by_name_zip', 'get_order_details', 'get_product_details', 'get_user_details', 'list_all_product_types', 'modify_pending_order_address', 'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address', 'return_delivered_order_items', 'think', 'transfer_to_human_agents']
import_statements = [f'from {i}.py import {i}\n' for i in allowed_functions]

analyse_codes(file_path, import_statements)
