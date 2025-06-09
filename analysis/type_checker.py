import ast
import os
from typing import Optional, Dict, Any, List
from code import Code


class TypeChecker(ast.NodeVisitor):
    def __init__(self, custom_funcs_folder="tau_bench/envs/retail/tools/"):
        # Map: function_name -> (param_types: Dict[str, str], required_params: List[str])
        self.funcs = {}
        self.load_custom_function_types(custom_funcs_folder)

        self.env = {}     # var name -> inferred type (str)
        self.errors = []  # List of error messages
        # print(self.funcs)
        # dlfkh

    def load_custom_function_types(self, folder: str):
        for filename in os.listdir(folder):
            if not filename.endswith('.py'):
                continue
            filepath = os.path.join(folder, filename)
            func_name, param_types, required = self.extract_get_info_types(filepath)
            if func_name:
                self.funcs[func_name] = (param_types, required)

    def extract_get_info_types(self, filepath: str):
        """
        Parses the file, finds get_info(), extracts function name, parameters and required parameters from returned dict.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_info':
                # Find the return statement
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        returned_dict = self.ast_literal_eval(stmt.value)
                        # returned_dict should be a dict now
                        try:
                            func_info = returned_dict.get("function", {})
                            func_name = func_info.get("name")
                            params = func_info.get("parameters", {})
                            properties = params.get("properties", {})
                            required = params.get("required", [])
                            param_types = {k: v.get("type", "Any") for k, v in properties.items()}
                            for param in param_types.keys():
                                if param_types[param] == "array":
                                    param_types[param] = "List"
                            return func_name, param_types, required
                        except Exception as e:
                            print(f"Error parsing get_info in {filepath}: {e}")
                            return None, None, None
        return None, None, None

    def ast_literal_eval(self, node: ast.AST) -> Any:
        """
        Like ast.literal_eval but supports nested dicts/lists of constants.
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Dict):
            return {self.ast_literal_eval(k): self.ast_literal_eval(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.List):
            return [self.ast_literal_eval(e) for e in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self.ast_literal_eval(e) for e in node.elts)
        else:
            raise ValueError(f"Unsupported AST node for literal eval: {ast.dump(node)}")

    def visit_Assign(self, node: ast.Assign):
        value_type = self.infer_type(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.env[target.id] = value_type
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        value_type = self.infer_type(node.value) if node.value else None
        if isinstance(node.target, ast.Name):
            self.env[node.target.id] = value_type
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name and func_name in self.funcs:
            param_types, required_params = self.funcs[func_name]

            arg_types = {}

            # match positional args
            param_names = list(param_types.keys())

            # positional args
            for i, arg in enumerate(node.args):
                if i >= len(param_names):
                    self.errors.append(f"Too many arguments in call to '{func_name}' at line {node.lineno}")
                    break
                arg_types[param_names[i]] = self.infer_type_str(arg)

            # keyword args
            for kw in node.keywords:
                arg_types[kw.arg] = self.infer_type_str(kw.value)

            # Check missing required params
            missing = [p for p in required_params if p not in arg_types]
            if missing:
                self.errors.append(f"Missing required arguments {missing} in call to '{func_name}' at line {node.lineno}")

            # Check types
            for param, expected in param_types.items():
                actual = arg_types.get(param)
                if actual is None:
                    continue  # Missing param error already reported above
                if not self.type_matches(actual, expected):
                    self.errors.append(
                        f"Type mismatch for parameter '{param}' in call to '{func_name}' at line {node.lineno}: expected '{expected}', got '{actual}'"
                    )

        self.generic_visit(node)

    def infer_type(self, node: ast.AST) -> Optional[str]:
        # Infer simplified type strings for literals or known vars
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.Num):  # Python <3.8
            return type(node.n).__name__
        elif isinstance(node, ast.Str):  # Python <3.8
            return 'str'
        elif isinstance(node, ast.Name):
            return self.env.get(node.id)
        elif isinstance(node, ast.Dict):
            return 'dict'
        elif isinstance(node, ast.List):
            return 'list'
        elif isinstance(node, ast.Tuple):
            return 'tuple'
        return None

    def infer_type_str(self, node: ast.AST) -> Optional[str]:
        # Wrapper to infer type as string for call argument checking
        return self.infer_type(node)

    def type_matches(self, actual: Optional[str], expected: Optional[str]) -> bool:
        if expected is None or expected.lower() == 'any':
            return True
        if actual is None:
            return True  # Can't determine actual type, skip error

        # Simplify expected and actual types (e.g. 'string' and 'str' => 'str')
        def normalize(t: str) -> str:
            t = t.lower()
            if t in ('string', 'str'):
                return 'str'
            if t == 'integer':
                return 'int'
            return t

        return normalize(actual) == normalize(expected)
    
    def analyze(self, code_obj: Code) -> List[str]:
        self.env.clear()
        self.errors.clear()
        self.visit(code_obj.code_ast)
        flag = len(self.errors)==0
        return flag, self.errors

    

