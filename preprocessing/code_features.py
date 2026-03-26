"""
Structural code feature extraction using regex-based analysis.
Extracts AST-like features, control flow patterns, algorithmic fingerprints,
opcode n-gram profiles, and AST subtree hashes from source code.
"""

import re
import hashlib
from collections import Counter
import numpy as np
from joblib import Parallel, delayed

# pycparser for AST subtree kernel (B3)
try:
    import pycparser
    _HAS_PYCPARSER = True
except ImportError:
    _HAS_PYCPARSER = False

# True LLVM IR extraction (requires clang installed)
# Falls back to pseudo-IR if not available
try:
    from utils.llvm_compiler import extract_true_llvm_ir
    _HAS_LLVM = True
except ImportError:
    _HAS_LLVM = False
    def extract_true_llvm_ir(code_str, timeout=5):
        return None

# ================= PRE-COMPILED REGEX PATTERNS =================

_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', flags=re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
_LOOP_RE = re.compile(r'\b(for|while|do)\s*[({]')
_BRANCH_RE = re.compile(r'\b(if|else|switch|case)\b')
_FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
_OPERATOR_RE = re.compile(r'[+\-*/%]')
_CF_RE = re.compile(r'\b(for|while|do|if|else|switch|case|return|break|continue)\b')

# TYPE-4 BEHAVIORAL FEATURES: These extract algorithm-level fingerprints
# that remain consistent across iterative vs recursive implementations.
_RETURN_RE = re.compile(r'\breturn\b')
_ACCUMULATOR_RE = re.compile(r'(\+=|-=|\*=|/=|%=|\b\w+\s*=\s*\w+\s*[+\-*/])')
_PARAM_RE = re.compile(r'\b\w+\s*\(([^)]*)\)')  # captures content inside first function parens
_MATH_OP_RE = re.compile(r'([+\-*/%])')  # individual math operators for skeleton

_FUNC_CALL_KEYWORDS = {
    'for', 'while', 'if', 'else', 'switch', 'case', 'return',
    'sizeof', 'typedef', 'do', 'defined', 'main'
}

# ================= A1: ALGORITHMIC FINGERPRINT PATTERNS =================

# Known library/standard functions in C/C++ (for library call set extraction)
_KNOWN_LIBRARY_FUNCS = {
    # I/O
    'printf', 'scanf', 'puts', 'gets', 'getchar', 'putchar',
    'fprintf', 'fscanf', 'fgets', 'fputs', 'fread', 'fwrite',
    'fopen', 'fclose', 'fflush', 'fseek', 'ftell', 'rewind',
    'sscanf', 'sprintf', 'snprintf',
    # Math
    'sqrt', 'pow', 'abs', 'fabs', 'ceil', 'floor', 'round',
    'log', 'log2', 'log10', 'exp', 'sin', 'cos', 'tan',
    'asin', 'acos', 'atan', 'atan2', 'hypot', 'fmod',
    # String
    'strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy', 'strcat',
    'strncat', 'strstr', 'strchr', 'strrchr', 'strtok', 'atoi',
    'atof', 'atol', 'strtol', 'strtod', 'tolower', 'toupper',
    'isdigit', 'isalpha', 'isalnum', 'isspace', 'isupper', 'islower',
    # Memory
    'malloc', 'calloc', 'realloc', 'free', 'memset', 'memcpy',
    'memmove', 'memcmp',
    # Sorting/searching
    'qsort', 'bsearch', 'sort', 'stable_sort', 'partial_sort',
    'lower_bound', 'upper_bound', 'binary_search', 'nth_element',
    'min_element', 'max_element', 'next_permutation', 'prev_permutation',
    # Container operations (C++)
    'push_back', 'pop_back', 'push_front', 'pop_front',
    'insert', 'erase', 'find', 'count', 'begin', 'end',
    'rbegin', 'rend', 'size', 'empty', 'clear', 'resize',
    'front', 'back', 'top', 'push', 'pop', 'swap',
    'make_pair', 'first', 'second',
    # Algorithm
    'min', 'max', 'accumulate', 'reverse', 'unique', 'fill',
    'copy', 'transform', 'for_each', 'count_if', 'find_if',
    # Utility
    'srand', 'rand', 'time', 'clock', 'exit', 'assert',
}

# Data structure detection patterns
_ARRAY_DECL_RE = re.compile(r'\b\w+\s*\[')       # array declaration/access
_LINKED_LIST_RE = re.compile(r'\b(struct\s+\w+\s*\*|->\s*next|->\s*prev|->\s*left|->\s*right|node\s*\*)', re.IGNORECASE)
_STACK_RE = re.compile(r'\b(stack|push|pop|top)\b')
_QUEUE_RE = re.compile(r'\b(queue|deque|priority_queue|front|back)\b')
_MAP_SET_RE = re.compile(r'\b(map|set|unordered_map|unordered_set|multimap|multiset)\s*<')
_VECTOR_RE = re.compile(r'\bvector\s*<')
_STRING_TYPE_RE = re.compile(r'\bstring\b')
_MATRIX_RE = re.compile(r'\w+\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]')  # 2D array

# IO pattern extraction
_IO_INPUT_RE = re.compile(r'\b(scanf|cin|getchar|gets|fgets|fscanf|fread)\b')
_IO_OUTPUT_RE = re.compile(r'\b(printf|cout|putchar|puts|fputs|fprintf|fwrite)\b')
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"].*?[>"]', re.MULTILINE)
_PREPROCESSOR_RE = re.compile(r'^\s*#.*$', re.MULTILINE)

# ================= PSEUDO-IR OPCODE EXTRACTORS =================
# Since clang cannot be guaranteed available on all host machines, we extract
# LLVM IR-like pseudo-opcodes from the source statically.

_IR_STORE_RE = re.compile(r'(?<![=!<>])=(?![=])')      # assignments
_IR_CMP_RE = re.compile(r'(==|!=|<=|>=|<|>)')           # icmp / fcmp
_IR_MATH_RE = re.compile(r'([+\-*/%])')                 # add, sub, mul, sdiv, srem
_IR_BITWISE_RE = re.compile(r'(&&|\|\||!|&|\||\^|~|<<|>>)') # and, or, xor, shl, ashr
_IR_ALLOC_RE = re.compile(r'\b(malloc|calloc|new|realloc|free|delete)\b') # allocations
_IR_VAR_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')  # generic variables (for load estimation)


# ================= A1: ALGORITHMIC FINGERPRINTING =================

def _extract_library_calls(code):
    """
    Extract the set of recognized library/standard functions used in code.
    Two codes solving the same problem often use similar library functions
    even if their structure is completely different (Type-4 signal).
    """
    calls = _FUNC_CALL_RE.findall(code)
    return frozenset(c for c in calls if c in _KNOWN_LIBRARY_FUNCS)


def _extract_data_structures(code):
    """
    Detect which data structure types are used in the code.
    Same-problem solutions often use similar data structures.
    Returns a frozenset of detected types.
    """
    structs = set()
    if _MATRIX_RE.search(code):
        structs.add('matrix')
    elif _ARRAY_DECL_RE.search(code):
        structs.add('array')
    if _LINKED_LIST_RE.search(code):
        structs.add('linked_list')
    if _STACK_RE.search(code):
        structs.add('stack')
    if _QUEUE_RE.search(code):
        structs.add('queue')
    if _MAP_SET_RE.search(code):
        structs.add('map_set')
    if _VECTOR_RE.search(code):
        structs.add('vector')
    if _STRING_TYPE_RE.search(code):
        structs.add('string')
    return frozenset(structs)


def _extract_io_pattern(code):
    """
    Extract a compact string encoding the I/O pattern of the code.
    Scans code line by line and records I (input) and O (output) operations.
    Same-problem solutions share similar I/O sequences even with different algorithms.
    Example: "IIIOIO" = read, read, read, write, read, write
    """
    pattern = []
    for line in code.splitlines():
        line_stripped = line.strip()
        has_input = bool(_IO_INPUT_RE.search(line_stripped))
        has_output = bool(_IO_OUTPUT_RE.search(line_stripped))
        if has_input:
            pattern.append('I')
        if has_output:
            pattern.append('O')
    return ''.join(pattern)


# ================= A2: N-GRAM OPCODE PROFILING =================

# Ordered mapping from source patterns to pseudo-opcodes
_OPCODE_PATTERNS = [
    (_IR_STORE_RE, 'store'),
    (_IR_CMP_RE, 'cmp'),
    (_IR_MATH_RE, 'math'),
    (_IR_BITWISE_RE, 'bitwise'),
    (_IR_ALLOC_RE, 'alloc'),
    (_LOOP_RE, 'br'),
    (_BRANCH_RE, 'br'),
    (_FUNC_CALL_RE, 'call'),
]


def _extract_opcode_sequence(code):
    """
    Extract an ordered sequence of pseudo-IR opcodes from the code.
    Unlike counting, this preserves the ORDER of operations,
    which is a stronger semantic signal for Type-4 clone detection.
    """
    # Build (position, opcode) tuples
    events = []
    for regex, opcode in _OPCODE_PATTERNS:
        for m in regex.finditer(code):
            events.append((m.start(), opcode))
    # Sort by position in source
    events.sort(key=lambda x: x[0])
    return [op for _, op in events]


def _compute_opcode_ngrams(opcode_seq, n=2):
    """
    Compute n-gram frequency profile from an opcode sequence.
    Returns a Counter of opcode n-grams.
    Example: ['store','cmp','math','br'] with n=2 →
             Counter({'store-cmp': 1, 'cmp-math': 1, 'math-br': 1})
    """
    if len(opcode_seq) < n:
        return Counter()
    ngrams = []
    for i in range(len(opcode_seq) - n + 1):
        ngrams.append('-'.join(opcode_seq[i:i+n]))
    return Counter(ngrams)


# ================= B3: AST SUBTREE KERNEL =================

def _strip_preprocessor(code):
    """
    Remove #include, #define and other preprocessor directives
    so pycparser can parse the code.
    """
    return _PREPROCESSOR_RE.sub('', code)


def _parse_ast_safe(code):
    """
    Attempt to parse code into a pycparser AST.
    Returns the AST root node or None if parsing fails.
    Strips preprocessor directives and handles common parse errors gracefully.
    """
    if not _HAS_PYCPARSER:
        return None
    try:
        clean = _strip_preprocessor(code)
        # Remove 'using namespace std;' and similar C++ that pycparser can't handle
        clean = re.sub(r'using\s+namespace\s+\w+\s*;', '', clean)
        # Replace cout/cin stream operators that pycparser can't handle
        clean = re.sub(r'\bcout\s*<<', 'printf(', clean)
        clean = re.sub(r'\bcin\s*>>', 'scanf(', clean)
        clean = re.sub(r'\bendl\b', '"\\n")', clean)
        parser = pycparser.CParser()
        ast = parser.parse(clean, filename='<code>')
        return ast
    except Exception:
        return None


def _hash_ast_node(node):
    """
    Compute a structural hash for an AST node (excluding identifiers/literals).
    This captures the "shape" of the code, not specific variable names.
    """
    if node is None:
        return 'None'
    node_type = node.__class__.__name__
    # Only use structure, not specific identifiers or values
    return node_type


def _extract_subtree_hashes(ast_node, max_depth=4):
    """
    Extract a set of structural subtree hashes from an AST.
    Uses a Weisfeiler-Lehman inspired approach: each subtree is hashed
    by combining its node type with the hashes of its children.
    
    max_depth limits the depth of subtrees to keep the feature space manageable.
    """
    if ast_node is None:
        return frozenset()
    
    subtree_hashes = set()
    
    def _wl_hash(node, depth):
        if depth > max_depth or node is None:
            return ''
        
        node_type = node.__class__.__name__
        
        # Get children
        children = node.children() if hasattr(node, 'children') else []
        if not children:
            h = node_type
            subtree_hashes.add(h)
            return h
        
        child_hashes = []
        for child_name, child_node in children:
            ch = _wl_hash(child_node, depth + 1)
            if ch:
                child_hashes.append(ch)
        
        # Sort child hashes for order-invariance within siblings
        child_hashes.sort()
        combined = f"{node_type}({','.join(child_hashes)})"
        
        # Hash long strings to keep them manageable
        if len(combined) > 64:
            combined = hashlib.md5(combined.encode()).hexdigest()[:16]
        
        subtree_hashes.add(combined)
        return combined
    
    _wl_hash(ast_node, 0)
    return frozenset(subtree_hashes)


# ================= COMMENT STRIPPING =================

def _strip_comments(code):
    """
    Remove C/C++ style comments from code.
    Handles both // line comments and /* block comments */.
    """
    code = _BLOCK_COMMENT_RE.sub('', code)
    code = _LINE_COMMENT_RE.sub('', code)
    return code


# ================= AST-LIKE FEATURES =================

def _count_loops(code):
    """Count loop statements (for, while, do)."""
    return len(_LOOP_RE.findall(code))


def _count_branches(code):
    """Count branching statements (if, else, switch, case)."""
    return len(_BRANCH_RE.findall(code))


def _count_func_calls(code):
    """Count function calls (identifier followed by parenthesis, excluding keywords)."""
    calls = _FUNC_CALL_RE.findall(code)
    return sum(1 for c in calls if c not in _FUNC_CALL_KEYWORDS)


def _compute_nesting_depth(code):
    """Compute maximum brace nesting depth."""
    max_depth = 0
    depth = 0
    for ch in code:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == '}':
            depth = max(0, depth - 1)
    return max_depth


def _count_operators(code):
    """Count arithmetic and comparison operators."""
    return len(_OPERATOR_RE.findall(code))


def _count_returns(code):
    """Count return statements."""
    return len(_RETURN_RE.findall(code))


def _detect_accumulator(code):
    """
    Detect accumulator patterns (+=, -=, x = x + y, etc.).
    Returns 1 if found, 0 otherwise.
    Both iterative (s += i) and recursive (return n + sum(n-1)) use accumulation.
    """
    # Direct accumulator operators
    if _ACCUMULATOR_RE.search(code):
        return 1
    # Recursive accumulation pattern: return <expr> + func(...)
    if re.search(r'return\s+\w+\s*[+\-*/]\s*\w+\s*\(', code):
        return 1
    return 0


def _count_params(code):
    """
    Count parameters in the first function definition.
    Iterative and recursive versions of the same algorithm
    typically have the same number of parameters.
    """
    # Match function definitions (not calls): type name(params)
    func_def = re.search(r'\b(?:int|void|float|double|long|char|bool|string|auto)\s+\w+\s*\(([^)]*)\)', code)
    if func_def:
        params = func_def.group(1).strip()
        if not params or params == 'void':
            return 0
        return len(params.split(','))
    return 0


def _math_op_set(code):
    """
    Extract the SET of unique math operators used.
    Both iterative sum and recursive sum use {+}, giving Jaccard=1.0.
    """
    return set(_MATH_OP_RE.findall(code))


# ================= CONTROL FLOW PATTERNS =================

_CF_MAPPING = {
    'for': 'F', 'while': 'W', 'do': 'D',
    'if': 'I', 'else': 'E',
    'switch': 'S', 'case': 'C',
    'return': 'R', 'break': 'B', 'continue': 'N'
}


def _extract_pseudo_ir_opcodes(code):
    """
    Extract frequency of pseudo LLVM-IR instructions.
    This gives the model a deeper semantic signature that is often shared 
    by Type-4 clones (e.g. iterative and recursive versions of the same algorithm
    often share underlying math/comparison instruction frequencies).
    """
    # Number of var usages roughly corresponds to 'load' instructions
    vars_found = _IR_VAR_RE.findall(code)
    # Exclude common keywords to only count actual identifiers
    load_count = sum(1 for v in vars_found if v not in _FUNC_CALL_KEYWORDS and v not in _CF_MAPPING)
    
    # Store: assignments
    store_count = len(_IR_STORE_RE.findall(code))
    
    # Cmp: logical comparisons
    cmp_count = len(_IR_CMP_RE.findall(code))
    
    # Math: arithmetic ops
    math_count = len(_IR_MATH_RE.findall(code))
    
    # Bitwise: logical and bitwise ops
    bitwise_count = len(_IR_BITWISE_RE.findall(code))
    
    # Branch: LLVM 'br' ops change instruction pointer.
    # Loops, conditions, and function calls (recursion) all represent branching.
    br_count = len(_LOOP_RE.findall(code)) + len(_BRANCH_RE.findall(code)) + len(_FUNC_CALL_RE.findall(code))
    
    # Allocations
    alloc_count = len(_IR_ALLOC_RE.findall(code))
    
    # Pseudo doesn't easily distinguish SSA phi and gep, so they default to 0.
    # Call is just the AST func call count.
    call_count = len(_FUNC_CALL_RE.findall(code))
    
    return [load_count, store_count, cmp_count, math_count, bitwise_count, br_count, alloc_count, 0, 0, call_count]




def _extract_cf_pattern(code):
    """
    Extract control flow pattern as a compact string signature.
    Example: for { if { } else { } return } → "FIERF"
    """
    tokens = _CF_RE.findall(code)
    return ''.join(_CF_MAPPING.get(t, '') for t in tokens)


def cf_pattern_similarity(pattern1, pattern2):
    """
    Compute similarity between two control flow patterns
    using bigram Jaccard similarity.
    """
    if not pattern1 and not pattern2:
        return 1.0
    if not pattern1 or not pattern2:
        return 0.0

    if len(pattern1) < 2 and len(pattern2) < 2:
        return 1.0 if pattern1 == pattern2 else 0.0

    def _bigrams(s):
        if len(s) < 2:
            return {s}
        return set(s[i:i+2] for i in range(len(s) - 1))

    bg1 = _bigrams(pattern1)
    bg2 = _bigrams(pattern2)

    union = len(bg1 | bg2)
    if union == 0:
        return 1.0
    return len(bg1 & bg2) / union


# ================= PUBLIC API =================

FEATURE_NAMES = [
    'branch_count', 'loop_call_combined',
    'nesting_depth', 'operator_count',
    # Algorithmic behavior features (TYPE-4 bridge)
    'return_count', 'accumulator_pattern', 'param_count', 'math_op_set_size',
    # pseudo/true IR features
    'ir_load_count', 'ir_store_count', 'ir_cmp_count',
    'ir_math_count', 'ir_bitwise_count', 'ir_br_count', 'ir_alloc_count',
    # True exclusive SSA/DataFlow IR features
    'ir_phi_count', 'ir_gep_count', 'ir_call_count',
    # A1: Algorithmic fingerprint numeric features
    'library_call_count', 'data_struct_count', 'io_pattern_length',
]


def _extract_single(code):
    """Extract features for a single code sample."""
    clean_code = _strip_comments(code)
    
    ast_feats = [
        _count_branches(clean_code),
        # loop_call_combined: loops + recursive calls = unified repetition metric
        _count_loops(clean_code) + _count_func_calls(clean_code),
        _compute_nesting_depth(clean_code),
        _count_operators(clean_code),
        # Algorithmic behavior features
        _count_returns(clean_code),
        _detect_accumulator(clean_code),
        _count_params(clean_code),
        len(_math_op_set(clean_code)),  # math_op_set_size
    ]
    
    # ATTEMPT TRUE LLVM IR
    true_ir_dict = extract_true_llvm_ir(code) # pass raw code, not clean_code, because compiler handles comments and needs standard raw syntax
    
    if true_ir_dict is not None:
        ir_feats = [
            true_ir_dict.get('load', 0),
            true_ir_dict.get('store', 0),
            true_ir_dict.get('cmp', 0),
            true_ir_dict.get('math', 0),
            true_ir_dict.get('bitwise', 0),
            true_ir_dict.get('br', 0),
            true_ir_dict.get('alloc', 0),
            true_ir_dict.get('phi', 0),
            true_ir_dict.get('gep', 0),
            true_ir_dict.get('call', 0)
        ]
    else:
        # FALLBACK TO PSEUDO-IR if code is not compilable
        ir_feats = _extract_pseudo_ir_opcodes(clean_code)
    
    # A1: Algorithmic fingerprint features
    lib_calls = _extract_library_calls(clean_code)
    data_structs = _extract_data_structures(clean_code)
    io_pattern = _extract_io_pattern(clean_code)
    math_ops = _math_op_set(clean_code)
    
    a1_numeric = [
        len(lib_calls),      # library_call_count
        len(data_structs),   # data_struct_count
        len(io_pattern),     # io_pattern_length
    ]
    
    # A2: Opcode sequence + n-grams
    opcode_seq = _extract_opcode_sequence(clean_code)
    opcode_ngrams = _compute_opcode_ngrams(opcode_seq, n=2)
    
    # B3: AST subtree hashes
    ast_tree = _parse_ast_safe(code)  # pass raw code, strip is done inside
    subtree_hashes = _extract_subtree_hashes(ast_tree, max_depth=4)
    
    feats = ast_feats + ir_feats + a1_numeric
    
    cf = _extract_cf_pattern(clean_code)
    
    # Semantic features dict (per-code, will be compared at pair level)
    semantic = {
        'library_calls': lib_calls,      # frozenset
        'data_structs': data_structs,     # frozenset
        'io_pattern': io_pattern,         # str
        'math_ops': math_ops,             # set
        'opcode_ngrams': opcode_ngrams,   # Counter
        'subtree_hashes': subtree_hashes, # frozenset
    }
    
    return feats, cf, semantic


def extract_all_features(raw_codes):
    """
    Extract per-code numeric features, control flow patterns,
    and semantic features for Type-4 clone detection.
    Parallelized with joblib for speed.

    Returns:
        features: np.ndarray of shape (n_codes, num_features)
        cf_patterns: list of control flow pattern strings
        semantic_features: dict with keys:
            'library_calls':   list of frozenset
            'data_structs':    list of frozenset
            'io_patterns':     list of str
            'math_ops':        list of set
            'opcode_ngrams':   list of Counter
            'subtree_hashes':  list of frozenset
    """
    print("Extracting code features (parallel)...")
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_extract_single)(code) for code in raw_codes
    )

    features = np.array([r[0] for r in results], dtype=np.float32)
    cf_patterns = [r[1] for r in results]
    
    # Aggregate semantic features into per-key lists
    semantic_features = {
        'library_calls':  [r[2]['library_calls'] for r in results],
        'data_structs':   [r[2]['data_structs'] for r in results],
        'io_patterns':    [r[2]['io_pattern'] for r in results],
        'math_ops':       [r[2]['math_ops'] for r in results],
        'opcode_ngrams':  [r[2]['opcode_ngrams'] for r in results],
        'subtree_hashes': [r[2]['subtree_hashes'] for r in results],
    }

    return features, cf_patterns, semantic_features
