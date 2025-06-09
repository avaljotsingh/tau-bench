import ast
import builtins
from termcolor import colored

from code import Code
from dead_code import DeadCodeAnalyzer
from type_checker import TypeChecker
from undefined_vars import UndefinedVariableAnalyzer

class StaticChecker:
    def __init__(self, code: Code):
        self.code = code

    def is_correct_syntax(self):
        try:
            ast.parse(self.code.code_str)
            return True
        except SyntaxError:
            return False
    
    def has_valid_function_calls(self, allowed_functions=['get_input_from_user']):
        allowed_functions = set(allowed_functions)
        builtin_funcs = set(dir(builtins))

        imported_modules = {}
        imported_functions = {}
        for node in ast.walk(self.code.code_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    imported_functions[alias.asname or alias.name] = module
        
        
        disallowed_calls = set()
        func_to_lib = {}
        
        def categorize(base_name, func_name, add_to_disallowed):
            if base_name in imported_functions:
                lib_name = imported_functions[func_name]
            elif func_name in builtin_funcs:
                lib_name = 'built-in'
            elif func_name in allowed_functions:
                lib_name = 'allowed custom'
            else:
                if add_to_disallowed:
                    lib_name = 'local or unknown'
                    disallowed_calls.add(func_name)
                else:
                    lib_name = 'method on variable or unknown'
            func_to_lib[func_name] = lib_name

        for node in ast.walk(self.code.code_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # e.g. foo()
                    func_name = node.func.id
                    categorize(func_name, func_name, True)
                
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
                        categorize(base_name, func_name, False)
        
        return len(disallowed_calls) == 0, disallowed_calls, func_to_lib
    
    def analyze_for(self, flag):
        if flag == 'syntax':
            if not self.is_correct_syntax():
                return False, 'The code generated is syntactically wrong'
        
        elif flag == 'valid_funcion_calls':
            flag, invalid_funcs, _ = self.has_valid_function_calls()
            if not flag:
                return False, f'The code generated has invalid function calls. The incorrect functions are {invalid_funcs}'
            
        elif flag == 'check_dead_code':
            analyzer = DeadCodeAnalyzer()
            flag, _, unused_vars = analyzer.analyze(self.code)
            if not flag:
                return False, str(unused_vars)
            
        elif flag == 'type_check':
            analyzer = TypeChecker()
            flag, errors = analyzer.analyze(self.code)
            if not flag:
                return flag, str(errors)
        
        elif flag == 'undefined_vars':
            analyzer = UndefinedVariableAnalyzer()
            flag, undefined_vars = analyzer.analyze(self.code)
            if not flag:
                return flag, f'The code generated uses undefined variables: {undefined_vars}'
            
        return True, None
    
    def modify(self, flag):
        if flag == 'remove_dead_code':
            analyzer = DeadCodeAnalyzer()
            self.code = analyzer.remove_dead_code(self.code)

    def analyze(self, properties):
        results = {}
        modifying_properties = ['remove_dead_code']
        for property in properties:
            if property in modifying_properties:
                self.modify(property)
            else:
                flag, comment = self.analyze_for(property)
                results[property] = (flag, comment)
                # if not flag:
                #     return flag, comment
        # return True, 'The code generated is correct'
        return results