import ast

allowed_functions = ['calculate', 'cancel_pending_order', 'exchange_delivered_order_items', 'find_user_id_by_email', 'find_user_id_by_name_zip', 'get_order_details', 'get_product_details', 'get_user_details', 'list_all_product_types', 'modify_pending_order_address', 'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address', 'return_delivered_order_items', 'think', 'transfer_to_human_agents', 'get_input_from_user']

class Code:
    import_statements = [f'from tau_bench.envs.retail.tools.{i} import {i}\n' for i in allowed_functions]
    
    def __init__(self, code_str, no_ast=False):
        self.code_str = code_str 
        self.no_ast = no_ast
        self.clean_code()
        self.add_imports()
        if not no_ast:
            self.code_ast = ast.parse(self.code_str)
        
    def clean_code(self):
        lines = self.code_str.strip().split('\n')
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
        lines = lines[len(Code.import_statements):]
        self.code_str = '\n'.join(lines)
        if not self.no_ast:
            self.code_ast = ast.parse(self.code_str)

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

    

    