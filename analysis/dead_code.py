import ast
import contextlib
import re
from code import Code

class DeadCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.assigned = set()
        self.used = set()
        self.assignments = {}
        self.dead_nodes = []
        self.unreachable_stack = [False]

    def is_unreachable(self):
        return self.unreachable_stack[-1]

    @contextlib.contextmanager
    def in_new_scope(self):
        self.unreachable_stack.append(False)
        try:
            yield
        finally:
            self.unreachable_stack.pop()

    def set_unreachable(self, value=True):
        self.unreachable_stack[-1] = value

    def mark_node_as_dead(self, node):
        self.dead_nodes.append(node)

    def visit_FunctionDef(self, node):
        with self.in_new_scope():
            self.generic_visit(node)

    def visit_If(self, node):
        with self.in_new_scope():
            self.visit(node.test)
            self.visit_statements(node.body)
        with self.in_new_scope():
            self.visit_statements(node.orelse)

    def visit_For(self, node):
        with self.in_new_scope():
            self.generic_visit(node)

    def visit_While(self, node):
        with self.in_new_scope():
            self.generic_visit(node)

    def visit_Try(self, node):
        with self.in_new_scope():
            self.visit_statements(node.body)
        for handler in node.handlers:
            with self.in_new_scope():
                self.visit(handler)
        with self.in_new_scope():
            self.visit_statements(node.finalbody)
        with self.in_new_scope():
            self.visit_statements(node.orelse)

    def visit_statements(self, stmts):
        for stmt in stmts:
            self.visit(stmt)

    def visit_Assign(self, node):
        if (
            len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
            and node.targets[0].id == node.value.id
        ):
            self.mark_node_as_dead(node)

        for target in node.targets:
            if isinstance(target, ast.Name):
                var = target.id
                self.assigned.add(var)
                self.assignments.setdefault(var, set()).add(node.lineno)

        if self.is_unreachable():
            self.mark_node_as_dead(node)

        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)

    def visit_Return(self, node):
        if self.is_unreachable():
            self.mark_node_as_dead(node)
        self.set_unreachable(True)
        self.generic_visit(node)

    def visit_Raise(self, node):
        if self.is_unreachable():
            self.mark_node_as_dead(node)
        self.set_unreachable(True)
        self.generic_visit(node)

    def visit_Break(self, node):
        if self.is_unreachable():
            self.mark_node_as_dead(node)
        self.set_unreachable(True)
        self.generic_visit(node)

    def visit_Expr(self, node):
        if self.is_unreachable():
            self.mark_node_as_dead(node)
        self.generic_visit(node)

    def generic_visit(self, node):
        if hasattr(node, 'lineno') and self.is_unreachable():
            if isinstance(node, (ast.Assign, ast.Expr, ast.AugAssign)):
                self.mark_node_as_dead(node)
        super().generic_visit(node)

    @staticmethod
    def insert_pass_in_empty_blocks(code_str):
        lines = code_str.splitlines()
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            block_header = re.match(r'^\s*(if|elif|else|for|while|try|except|finally)\b.*:\s*$', line)
            if block_header:
                indent = len(line) - len(line.lstrip())
                j = i + 1
                has_code = False
                while j < len(lines):
                    next_line = lines[j]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= indent:
                        break
                    if next_line.strip() and not next_line.strip().startswith("#"):
                        has_code = True
                        break
                    j += 1
                if not has_code:
                    new_lines.append(" " * (indent + 4) + "pass")
            i += 1
        return '\n'.join(new_lines)

    def analyze(self, code_obj):
        self.visit(code_obj.code_ast)
        unused_vars = self.assigned - self.used
        for var in unused_vars:
            for lineno in self.assignments.get(var, []):
                for node in ast.walk(code_obj.code_ast):
                    if isinstance(node, ast.Assign) and node.lineno == lineno:
                        self.mark_node_as_dead(node)

        all_dead = len(self.dead_nodes) == 0
        return all_dead, self.dead_nodes, unused_vars

    def remove_dead_code(self, code_obj):
        flag, dead_nodes, _ = self.analyze(code_obj)
        if flag:
            return code_obj

        dead_lines = set()
        for node in dead_nodes:
            start = node.lineno
            end = getattr(node, 'end_lineno', start)
            dead_lines.update(range(start, end + 1))

        code_lines = code_obj.code_str.splitlines()
        new_code_lines = [
            line for i, line in enumerate(code_lines, start=1)
            if i not in dead_lines
        ]
        new_code_str = self.insert_pass_in_empty_blocks('\n'.join(new_code_lines))
        new_code_obj = Code(new_code_str)
        return new_code_obj
