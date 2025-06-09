import ast
import contextlib
import builtins
from code import Code

class UndefinedVariableAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.scopes = [{}]  # Stack of dicts: var_name -> defined (True/False)
        self.used_before_def = []
        self.defined_globals = set()
        self.builtins = set(dir(builtins))

    @contextlib.contextmanager
    def new_scope(self):
        self.scopes.append({})
        try:
            yield
        finally:
            self.scopes.pop()

    def define(self, name):
        self.scopes[-1][name] = True
        self.defined_globals.add(name)

    def is_defined(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return True
        return name in self.builtins

    def visit_FunctionDef(self, node):
        self.define(node.name)
        with self.new_scope():
            for arg in node.args.args:
                self.define(arg.arg)
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.define(node.name)
        with self.new_scope():
            self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.define(target.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            if not self.is_defined(node.target.id):
                self.used_before_def.append((node.target.id, node.lineno))
            self.define(node.target.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not self.is_defined(node.id):
                self.used_before_def.append((node.id, node.lineno))
        elif isinstance(node.ctx, ast.Store):
            self.define(node.id)

    def visit_Import(self, node):
        for alias in node.names:
            self.define(alias.asname or alias.name.split('.')[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.define(alias.asname or alias.name)

    def visit_comprehension(self, comp):
        # comp.target is where the loop variable is introduced
        if isinstance(comp.target, ast.Name):
            self.define(comp.target.id)
        self.visit(comp.iter)
        for if_clause in comp.ifs:
            self.visit(if_clause)

    def visit_ListComp(self, node):
        with self.new_scope():
            for comp in node.generators:
                self.visit_comprehension(comp)
            self.visit(node.elt)

    def visit_SetComp(self, node):
        with self.new_scope():
            for comp in node.generators:
                self.visit_comprehension(comp)
            self.visit(node.elt)

    def visit_DictComp(self, node):
        with self.new_scope():
            for comp in node.generators:
                self.visit_comprehension(comp)
            self.visit(node.key)
            self.visit(node.value)

    def visit_GeneratorExp(self, node):
        with self.new_scope():
            for comp in node.generators:
                self.visit_comprehension(comp)
            self.visit(node.elt)

    def analyze(self, code_obj):
        self.visit(code_obj.code_ast)
        flag = len(self.used_before_def)==0
        return flag, self.used_before_def
