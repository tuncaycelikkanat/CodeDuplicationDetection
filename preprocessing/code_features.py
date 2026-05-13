"""
Structural code feature extraction using regex-based analysis.
Extracts basic code structure counters, control flow patterns, and algorithmic fingerprints.
Simplified to focus strictly on features proven useful for semantic (Type-4) and near-miss clones.
"""

import re
from collections import Counter
import numpy as np
from joblib import Parallel, delayed
from preprocessing.tree_sitter_parser import get_parser

# ================= PRE-COMPILED REGEX PATTERNS =================

_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', flags=re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
_LOOP_RE = re.compile(r'\b(for|while|do)\s*[({]')
_BRANCH_RE = re.compile(r'\b(if|else|switch|case)\b')
_FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
_OPERATOR_RE = re.compile(r'[+\-*/%]')
_CF_RE = re.compile(r'\b(for|while|do|if|else|switch|case|return|break|continue)\b')

# Behavioral fingerprint regexes
_RETURN_RE = re.compile(r'\breturn\b')
_ACCUMULATOR_RE = re.compile(r'(\+=|-=|\*=|/=|%=|\b\w+\s*=\s*\w+\s*[+\-*/])')
_PARAM_RE = re.compile(r'\b\w+\s*\(([^)]*)\)')
_MATH_OP_RE = re.compile(r'([+\-*/%])')

_FUNC_CALL_KEYWORDS = {
    'for', 'while', 'if', 'else', 'switch', 'case', 'return',
    'sizeof', 'typedef', 'do', 'defined', 'main'
}

# Known library/standard functions in C/C++
_KNOWN_LIBRARY_FUNCS = {
    'printf', 'scanf', 'puts', 'gets', 'getchar', 'putchar', 'fprintf', 'fscanf', 'fgets', 'fputs', 'fread', 'fwrite',
    'fopen', 'fclose', 'fflush', 'fseek', 'ftell', 'rewind', 'sscanf', 'sprintf', 'snprintf',
    'sqrt', 'pow', 'abs', 'fabs', 'ceil', 'floor', 'round', 'log', 'log2', 'log10', 'exp', 'sin', 'cos', 'tan',
    'strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy', 'strcat', 'strstr', 'strchr', 'strrchr', 'strtok', 'atoi',
    'malloc', 'calloc', 'realloc', 'free', 'memset', 'memcpy', 'memmove', 'memcmp',
    'qsort', 'bsearch', 'sort', 'stable_sort', 'binary_search',
    'push_back', 'pop_back', 'insert', 'erase', 'find', 'count', 'size', 'empty', 'clear', 'push', 'pop', 'swap',
    'min', 'max', 'accumulate', 'reverse', 'fill', 'copy',
    'srand', 'rand', 'time', 'clock', 'exit', 'assert', 'cin', 'cout', 'new', 'delete'
}

_LIB_CATEGORIES = {
    'IO': frozenset({'printf', 'scanf', 'puts', 'gets', 'getchar', 'putchar', 'fprintf', 'fscanf', 'fgets', 'fputs', 'fread', 'fwrite', 'cin', 'cout'}),
    'FILE': frozenset({'fopen', 'fclose', 'fflush', 'fseek', 'ftell', 'rewind'}),
    'MATH': frozenset({'sqrt', 'pow', 'abs', 'fabs', 'ceil', 'floor', 'round', 'log', 'log2', 'log10', 'exp', 'sin', 'cos', 'tan'}),
    'STRING': frozenset({'strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy', 'strcat', 'strstr', 'strchr', 'strrchr', 'strtok', 'atoi'}),
    'MEMORY': frozenset({'malloc', 'calloc', 'realloc', 'free', 'memset', 'memcpy', 'memmove', 'memcmp', 'new', 'delete'}),
    'ALGO': frozenset({'qsort', 'bsearch', 'sort', 'stable_sort', 'binary_search', 'accumulate', 'reverse', 'fill', 'copy', 'swap', 'min', 'max'}),
    'SYS': frozenset({'srand', 'rand', 'time', 'clock', 'exit', 'assert'})
}

# Data structure detection patterns
_ARRAY_DECL_RE = re.compile(r'\b\w+\s*\[')
_LINKED_LIST_RE = re.compile(r'\b(struct\s+\w+\s*\*|->\s*next|->\s*prev|->\s*left|->\s*right|node\s*\*)', re.IGNORECASE)
_STACK_RE = re.compile(r'\b(stack|push|pop|top)\b')
_QUEUE_RE = re.compile(r'\b(queue|deque|priority_queue|front|back)\b')
_MAP_SET_RE = re.compile(r'\b(map|set|unordered_map|unordered_set|multimap|multiset)\s*<')
_VECTOR_RE = re.compile(r'\bvector\s*<')
_STRING_TYPE_RE = re.compile(r'\bstring\b')
_MATRIX_RE = re.compile(r'\w+\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]')

# IO pattern extraction
_IO_INPUT_RE = re.compile(r'\b(scanf|cin|getchar|gets|fgets|fscanf|fread)\b')
_IO_OUTPUT_RE = re.compile(r'\b(printf|cout|putchar|puts|fputs|fprintf|fwrite)\b')

# Pure ML Features (Halstead, McCabe, Skeleton, Type Profile)
_HALSTEAD_OP_RE = re.compile(r'(\+|-|\*|/|%|=|==|!=|<|>|<=|>=|&&|\|\||!|&|\||\^|<<|>>|if|else|for|while|return|switch|case)')
_HALSTEAD_OPERAND_RE = re.compile(r'\b([a-zA-Z_]\w*|\d+\.?\d*)\b')
_MCCABE_RE = re.compile(r'\b(if|while|for|case)\b|&&|\|\||\?')
_SKELETON_RE = re.compile(r'\b(if|else|for|while|do|switch|case|return|break|continue)\b|[+\-*/%<>=!&|^~?]')

_TYPE_INT = re.compile(r'\b(int|long|short|unsigned|signed)\b')
_TYPE_FLOAT = re.compile(r'\b(float|double)\b')
_TYPE_CHAR = re.compile(r'\b(char|string)\b')
_TYPE_STRUCT = re.compile(r'\b(struct|class)\b')
_PTR = re.compile(r'\*')
_ARRAY = re.compile(r'\[')

_ARRAY_ACCESS_RE = re.compile(r'\b[a-zA-Z_]\w*\s*\[')
_PTR_DEREF_RE = re.compile(r'\*[a-zA-Z_]\w*')


# ================= ALGORITHMIC FINGERPRINTING =================

def _extract_library_calls(code):
    calls = _FUNC_CALL_RE.findall(code)
    return frozenset(c for c in calls if c in _KNOWN_LIBRARY_FUNCS)

def _extract_library_categories(calls):
    categories = set()
    for call in calls:
        for cat, funcs in _LIB_CATEGORIES.items():
            if call in funcs:
                categories.add(cat)
    return frozenset(categories)

def _extract_data_structures(code):
    structs = set()
    if _MATRIX_RE.search(code): structs.add('matrix')
    elif _ARRAY_DECL_RE.search(code): structs.add('array')
    if _LINKED_LIST_RE.search(code): structs.add('linked_list')
    if _STACK_RE.search(code): structs.add('stack')
    if _QUEUE_RE.search(code): structs.add('queue')
    if _MAP_SET_RE.search(code): structs.add('map_set')
    if _VECTOR_RE.search(code): structs.add('vector')
    if _STRING_TYPE_RE.search(code): structs.add('string')
    return frozenset(structs)

def _extract_io_pattern(code):
    pattern = []
    for line in code.splitlines():
        line_stripped = line.strip()
        if _IO_INPUT_RE.search(line_stripped): pattern.append('I')
        if _IO_OUTPUT_RE.search(line_stripped): pattern.append('O')
    return ''.join(pattern)

def _math_op_set(code):
    return set(_MATH_OP_RE.findall(code))

# ================= CODE METRICS & COUNTERS =================

def _strip_comments(code):
    code = _BLOCK_COMMENT_RE.sub('', code)
    code = _LINE_COMMENT_RE.sub('', code)
    return code

def _count_loops(code): return len(_LOOP_RE.findall(code))
def _count_branches(code): return len(_BRANCH_RE.findall(code))
def _count_func_calls(code):
    calls = _FUNC_CALL_RE.findall(code)
    return sum(1 for c in calls if c not in _FUNC_CALL_KEYWORDS)
def _count_operators(code): return len(_OPERATOR_RE.findall(code))
def _count_returns(code): return len(_RETURN_RE.findall(code))

def _count_array_accesses(code): return len(_ARRAY_ACCESS_RE.findall(code))
def _count_ptr_derefs(code): return len(_PTR_DEREF_RE.findall(code))

def _compute_nesting_depth(code):
    max_depth, depth = 0, 0
    for ch in code:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == '}':
            depth = max(0, depth - 1)
    return max_depth

def _detect_accumulator(code):
    if _ACCUMULATOR_RE.search(code): return 1
    if re.search(r'return\s+\w+\s*[+\-*/]\s*\w+\s*\(', code): return 1
    return 0

def _count_params(code):
    func_def = re.search(r'\b(?:int|void|float|double|long|char|bool|string|auto)\s+\w+\s*\(([^)]*)\)', code)
    if func_def:
        params = func_def.group(1).strip()
        if not params or params == 'void': return 0
        return len(params.split(','))
    return 0

def _compute_halstead(code):
    operators = _HALSTEAD_OP_RE.findall(code)
    operands = [w for w in _HALSTEAD_OPERAND_RE.findall(code) if w not in _FUNC_CALL_KEYWORDS]
    
    n1 = len(set(operators))
    n2 = len(set(operands))
    N1 = len(operators)
    N2 = len(operands)
    
    vocab = n1 + n2
    length = N1 + N2
    
    if vocab == 0:
        return 0.0, 0.0
        
    volume = length * np.log2(vocab)
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    effort = difficulty * volume
    return float(volume), float(effort)

def _compute_mccabe(code):
    return len(_MCCABE_RE.findall(code)) + 1

def _extract_skeleton(code):
    return tuple(_SKELETON_RE.findall(code))

def _type_profile(code):
    return np.array([
        len(_TYPE_INT.findall(code)),
        len(_TYPE_FLOAT.findall(code)),
        len(_TYPE_CHAR.findall(code)),
        len(_TYPE_STRUCT.findall(code)),
        len(_PTR.findall(code)),
        len(_ARRAY.findall(code))
    ], dtype=np.float32)

# ================= CONTROL FLOW PATTERNS =================

_CF_MAPPING = {
    'for': 'F', 'while': 'W', 'do': 'D', 'if': 'I', 'else': 'E',
    'switch': 'S', 'case': 'C', 'return': 'R', 'break': 'B', 'continue': 'N'
}

_ABSTRACT_CF_MAPPING = {
    'for': 'L', 'while': 'L', 'do': 'L',
    'if': 'B', 'else': 'B', 'switch': 'B', 'case': 'B',
    'return': 'R', 'break': 'J', 'continue': 'J'
}

def _extract_cf_pattern(code):
    tokens = _CF_RE.findall(code)
    return ''.join(_CF_MAPPING.get(t, '') for t in tokens)

def _extract_abstract_cf_pattern(code):
    tokens = _CF_RE.findall(code)
    return ''.join(_ABSTRACT_CF_MAPPING.get(t, '') for t in tokens)

def cf_pattern_similarity(pattern1, pattern2):
    if not pattern1 and not pattern2: return 1.0
    if not pattern1 or not pattern2: return 0.0
    if len(pattern1) < 2 and len(pattern2) < 2: return 1.0 if pattern1 == pattern2 else 0.0

    def _bigrams(s):
        if len(s) < 2: return {s}
        return set(s[i:i+2] for i in range(len(s) - 1))

    bg1 = _bigrams(pattern1)
    bg2 = _bigrams(pattern2)
    union = len(bg1 | bg2)
    return len(bg1 & bg2) / union if union > 0 else 1.0

# ================= PUBLIC API =================

FEATURE_NAMES = [
    'branch_count', 'loop_call_combined', 'nesting_depth', 'operator_count',
    'return_count', 'accumulator_pattern', 'param_count', 'math_op_set_size',
    'library_call_count', 'data_struct_count', 'io_pattern_length',
    'halstead_volume', 'halstead_effort', 'mccabe_complexity',
    'array_access_count', 'ptr_deref_count',
    # Normalized density metrics (robust to code length variation)
    'operator_density', 'branch_density', 'loop_density', 'halstead_vol_per_line'
]

def _extract_single(code):
    clean_code = _strip_comments(code)
    parser = get_parser()
    tree = parser.parse(clean_code)
    
    # Extract numeric features via Tree-sitter
    loops = parser.count_matches(tree, 'loops')
    func_calls = len(parser.extract_func_calls(tree, clean_code))
    branches = parser.count_matches(tree, 'branches')
    operators = parser.count_matches(tree, 'operators')
    returns = parser.count_matches(tree, 'returns')
    params = parser.count_matches(tree, 'params')
    math_ops = parser.extract_math_ops(tree)
    
    feats = [
        branches,
        loops + func_calls,
        _compute_nesting_depth(clean_code),
        operators,
        returns,
        _detect_accumulator(clean_code),
        params,
        len(math_ops),
    ]
    
    # Semantic & Algorithmic behaviors (mixed regex/tree-sitter)
    lib_calls = _extract_library_calls(clean_code)
    data_structs = _extract_data_structures(clean_code)
    io_pattern = _extract_io_pattern(clean_code)
    
    # Pure ML Structural Metrics
    h_vol, h_eff = _compute_halstead(clean_code)
    mccabe = _compute_mccabe(clean_code)

    feats.extend([
        len(lib_calls),
        len(data_structs),
        len(io_pattern),
        h_vol,
        h_eff,
        mccabe,
        parser.count_matches(tree, 'array_access'),
        parser.count_matches(tree, 'ptr_deref')
    ])
    
    # ---- Normalized density metrics (length-invariant, better for Type-4) ----
    num_lines = max(len(clean_code.splitlines()), 1)
    num_tokens = max(len(clean_code.split()), 1)
    feats.append(feats[3] / num_tokens)   # operator_density = operators / tokens
    feats.append(feats[0] / num_lines)    # branch_density   = branches / lines
    feats.append(feats[1] / num_lines)    # loop_density     = loops / lines
    feats.append(h_vol / num_lines)       # halstead_vol_per_line

    cf = parser.extract_control_flow(tree)
    
    semantic = {
        'library_calls': lib_calls,
        'library_categories': _extract_library_categories(lib_calls),
        'data_structs': data_structs,
        'io_pattern': io_pattern,
        'math_ops': math_ops,
        'skeleton': _extract_skeleton(clean_code),
        'type_profile': _type_profile(clean_code),
        'abstract_cf': cf
    }
    
    return feats, cf, semantic

def extract_all_features(raw_codes):
    print("Extracting simplified code features (parallel)...")
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_extract_single)(code) for code in raw_codes
    )

    features = np.array([r[0] for r in results], dtype=np.float32)
    cf_patterns = [r[1] for r in results]
    
    semantic_features = {
        'library_calls':  [r[2]['library_calls'] for r in results],
        'library_categories': [r[2]['library_categories'] for r in results],
        'data_structs':   [r[2]['data_structs'] for r in results],
        'io_patterns':    [r[2]['io_pattern'] for r in results],
        'math_ops':       [r[2]['math_ops'] for r in results],
        'skeletons':      [r[2]['skeleton'] for r in results],
        'type_profiles':  [r[2]['type_profile'] for r in results],
        'abstract_cf_patterns': [r[2]['abstract_cf'] for r in results],
    }

    return features, cf_patterns, semantic_features
