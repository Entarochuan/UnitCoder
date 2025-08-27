from typing import List, Dict, Optional, Tuple, Set
import ast
import astor
import subprocess

def source_to_ast(code: str) -> ast.AST:
    try:
        return ast.parse(code)
    except Exception as e:
        if not isinstance(e, SyntaxError):
            print(f'{type(e).__name__}: {e}')
        return None

def get_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]

def ast_to_source(node: ast.AST) -> str:
    try:
        return astor.to_source(node)
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return ''

def has_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
    return False

def get_used_names_and_attrs(node: ast.AST) -> set:
    used = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            used.add(n.attr)
            if isinstance(n.value, ast.Name):
                used.add(n.value.id)
    return used


def get_imports(tree: ast.AST) -> Tuple[Dict[str, str], List[str]]:
    imports_dict = {}
    original_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_str = ast_to_source(node)
            original_imports.append(import_str)
            for alias in node.names:
                imports_dict[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            import_str = ast_to_source(node)
            original_imports.append(import_str)
            module = node.module or ''
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                imports_dict[alias.asname or alias.name] = full_name
    return imports_dict, original_imports

        
def get_used_names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

class FunctionAst:
    def __init__(self, node: ast.FunctionDef):
        self.node = node
        assert isinstance(node, ast.FunctionDef)
        
    @property
    def name(self):
        return self.node.name
    
    @property
    def args(self):
        return [arg.arg for arg in self.node.args.args]
    
    def is_nested_function(self):  # 检测嵌套函数
        for node in ast.iter_child_nodes(self.node):
            if isinstance(node, ast.FunctionDef):
                return True
        return False
    
    def has_return(self):
        for node in ast.walk(self.node):
            if isinstance(node, ast.Return):
                return True
        return False
    
    def has_args(self):
        return len(self.args) > 0
    
    def has_import(self):
        for node in ast.walk(self.node):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return True
        return False
    
