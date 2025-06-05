import ast

class DeadCodeDetector(ast.NodeVisitor):
    def __init__(self):
        self.assigned = set()
        self.used = set()
        self.dead_lines = set()
        self.in_function = False
        self.unreachable = False
    
    def visit_FunctionDef(self, node):
        self.in_function = True
        self.unreachable = False
        self.generic_visit(node)
        self.in_function = False
    
    def visit_Assign(self, node):
        # Check for no-op assignment: e.g., x = x
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
            isinstance(node.value, ast.Name) and node.targets[0].id == node.value.id):
            self.dead_lines.add(node.lineno)
        
        # Track assigned vars
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned.add(target.id)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
    
    def visit_Return(self, node):
        # After return, next statements are unreachable
        self.unreachable = True
        self.generic_visit(node)
    
    def visit_Raise(self, node):
        self.unreachable = True
        self.generic_visit(node)
    
    def visit_Break(self, node):
        self.unreachable = True
        self.generic_visit(node)
    
    def generic_visit(self, node):
        if self.unreachable and hasattr(node, 'lineno'):
            # Mark unreachable lines
            self.dead_lines.add(node.lineno)
            # But keep visiting to find all unreachable lines
        super().generic_visit(node)


def no_dead_code(ast_tree):
    detector = DeadCodeDetector()
    detector.visit(ast_tree)
    # Variables assigned but never used
    unused_vars = detector.assigned - detector.used

    flag = len(unused_vars)==0 and len(detector.dead_lines)==0
    return flag, sorted(detector.dead_lines), unused_vars

# Example usage:
if __name__ == "__main__":
    sample_code = '''
def foo():
    x = 5
    y = x
    x = x  # no-op
    return y
    print("unreachable")

z = 10  # assigned but never used
'''
    tree = ast.parse(sample_code)
    result = no_dead_code(tree)
    print("Dead lines:", result['dead_lines'])
    print("Unused variables:", result['unused_variables'])
