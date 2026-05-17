import tree_sitter_cpp
from tree_sitter import Language, Parser, Query, QueryCursor

# Singleton Language initialization
CPP_LANGUAGE = Language(tree_sitter_cpp.language())

# Pre-compile common queries for performance
QUERIES = {
    'loops': Query(CPP_LANGUAGE, """
        (for_statement) @loop
        (while_statement) @loop
        (do_statement) @loop
    """),
    'branches': Query(CPP_LANGUAGE, """
        (if_statement) @branch
        (switch_statement) @branch
        (case_statement) @branch
    """),
    'func_calls': Query(CPP_LANGUAGE, """
        (call_expression
            function: (identifier) @func_name
        )
    """),
    'operators': Query(CPP_LANGUAGE, """
        (binary_expression operator: _ @op)
        (update_expression operator: _ @op)
        (assignment_expression operator: _ @op)
    """),
    'returns': Query(CPP_LANGUAGE, """
        (return_statement) @ret
    """),
    'math_ops': Query(CPP_LANGUAGE, """
        (binary_expression operator: ["+" "-" "*" "/" "%"] @math)
        (assignment_expression operator: ["+=" "-=" "*=" "/=" "%="] @math)
        (update_expression operator: ["++" "--"] @math)
    """),
    'array_access': Query(CPP_LANGUAGE, """
        (subscript_expression) @arr
    """),
    'ptr_deref': Query(CPP_LANGUAGE, """
        (pointer_expression) @ptr
    """),
    'params': Query(CPP_LANGUAGE, """
        (parameter_declaration) @param
        (optional_parameter_declaration) @param
    """),
    # For abstract CF (Control Flow) extraction
    'control_flow': Query(CPP_LANGUAGE, """
        (for_statement) @cf_loop
        (while_statement) @cf_loop
        (do_statement) @cf_loop
        (if_statement) @cf_branch
        (switch_statement) @cf_branch
        (case_statement) @cf_branch
        (return_statement) @cf_return
        (break_statement) @cf_break
        (continue_statement) @cf_continue
    """)
}

class ASTParser:
    def __init__(self):
        self.parser = Parser(CPP_LANGUAGE)

    def parse(self, code_str):
        if not code_str:
            return None
        # UTF-8 encoded bytes for tree-sitter
        return self.parser.parse(bytes(code_str, "utf8"))

    def count_matches(self, tree, query_name):
        if not tree: return 0
        query = QUERIES.get(query_name)
        if not query: return 0
        
        cursor = QueryCursor(query)
        # matches returns a generator of tuples: (pattern_index, captures)
        matches = list(cursor.matches(tree.root_node))
        return len(matches)

    def extract_func_calls(self, tree, code_bytes):
        if not tree: return set()
        query = QUERIES['func_calls']
        cursor = QueryCursor(query)
        
        func_names = []
        for match in cursor.matches(tree.root_node):
            captures = match[1]
            for name, nodes_list in captures.items():
                if name == "func_name":
                    for node in nodes_list:
                        func_names.append(node.text.decode("utf8"))
        return set(func_names)

    def extract_control_flow(self, tree):
        """Extracts the control flow skeleton in order of appearance."""
        if not tree: return "", ""
        
        query = QUERIES['control_flow']
        cursor = QueryCursor(query)
        
        # We need to sort by start_byte to get the exact sequence in the code
        nodes = []
        for match in cursor.matches(tree.root_node):
            captures = match[1]
            for name, nodes_list in captures.items():
                for node in nodes_list:
                    nodes.append((node.start_byte, name))
                
        nodes.sort(key=lambda x: x[0])
        
        # CF Mapping
        cf_map = {
            'cf_loop': 'L',
            'cf_branch': 'B',
            'cf_return': 'R',
            'cf_break': 'J',
            'cf_continue': 'J'
        }
        
        pattern = "".join([cf_map.get(name, "?") for _, name in nodes])
        return pattern

    def extract_math_ops(self, tree):
        if not tree: return set()
        query = QUERIES['math_ops']
        cursor = QueryCursor(query)
        
        ops = set()
        for match in cursor.matches(tree.root_node):
            captures = match[1]
            for name, nodes_list in captures.items():
                for node in nodes_list:
                    ops.add(node.text.decode("utf8"))
        return ops

    def compute_max_depth(self, tree):
        """
        Computes the maximum nesting depth of the AST by traversing scope-inducing nodes.
        This safely replaces the old text-based '{' counting mechanism, 
        ignoring strings, comments, and properly counting scopes without braces.
        """
        if not tree: return 0
        
        # Nodes that increase logical depth
        SCOPE_NODES = {
            'compound_statement', 'for_statement', 'while_statement', 
            'do_statement', 'if_statement', 'switch_statement'
        }
        
        def _get_depth(node):
            if not node.children:
                return 0
            
            max_child_depth = max((_get_depth(c) for c in node.children), default=0)
            return max_child_depth + 1 if node.type in SCOPE_NODES else max_child_depth
            
        return _get_depth(tree.root_node)

# Singleton instance for lazy loading (to support Joblib Multiprocessing)
_ast_parser_instance = None

def get_parser():
    global _ast_parser_instance
    if _ast_parser_instance is None:
        _ast_parser_instance = ASTParser()
    return _ast_parser_instance
