import ast

allowed_functions = ['calculate', 'cancel_pending_order', 'exchange_delivered_order_items', 'find_user_id_by_email', 'find_user_id_by_name_zip', 'get_order_details', 'get_product_details', 'get_user_details', 'list_all_product_types', 'modify_pending_order_address', 'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address', 'return_delivered_order_items', 'think', 'transfer_to_human_agents', 'get_input_from_user']

def parse_code(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        print(code)
        return None

class Code:
    import_statements = [f'from tau_bench.envs.retail.tools.{i} import {i}\n' for i in allowed_functions]
    
    def __init__(self, code_str, no_ast=False):
        self.code_str = code_str 
        self.no_ast = no_ast
        self.clean_code()
        self.remove_imports()
        self.add_imports()
        if not no_ast:
            self.code_ast = parse_code(self.code_str)
        
    def clean_code(self):
        lines = self.code_str.strip().split('\n')
        new_lines = []
        flag = False
        for line in lines:
            if line.startswith("```"):
                flag = True
            if flag:
                new_lines.append(line)
            if line.endswith("```"):
                flag = False
        if len(new_lines) == 0:
            new_lines = lines
        lines = new_lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        self.code_str = '\n'.join(lines)
        # self.remove_comments()
        lines = self.code_str.splitlines()
        new_lines = []
        for line in lines:
            if line.strip():
                new_lines.append(line)
        self.code_str = "\n".join(new_lines)

    def add_imports(self):
        self.code_str = ''.join(Code.import_statements) + self.code_str

    def remove_imports(self):
        lines = self.code_str.splitlines()
        new_lines = []
        for line in lines:
            if 'import' not in line:
                new_lines.append(line)
        self.code_str = '\n'.join(new_lines)
        # lines = lines[len(Code.import_statements):]
        # self.code_str = '\n'.join(lines)
        if not self.no_ast:
            self.code_ast = parse_code(self.code_str)

    def remove_comments(self):
        lines = self.code_str.splitlines()
        new_lines = []
        for line in lines:
            if line.strip():
                if line.startswith('#'):
                    continue 
                new_lines.append(line)
        self.code_str = '\n'.join(new_lines)

    def add_line_numbers(self, code_str):
        lines = code_str.splitlines()
        numbered_lines = []
        line_num = 1
        for line in lines:
            if line.strip(): 
                numbered_lines.append(f"{line_num:3d}: {line}")
                line_num += 1
        return "\n".join(numbered_lines)
        

    def pretty_print(self, line_numbers=True):
        if line_numbers:
            print(self.add_line_numbers(self.code_str))
        else:
            print(self.code_str)
        print()

    def add_hash(self):
        lines = self.code_str.splitlines()
        new_lines = []
        for line in lines:
            if "order_id=" in line:
                line = line.replace('order_id="', 'order_id = "#')
            new_lines.append(line)
        self.code_str = '\n'.join(new_lines)
        if not self.no_ast:
            self.code_ast = parse_code(self.code_str)
        return self

    

    