"""
Structural code feature extraction using regex-based analysis.
Extracts AST-like features and control flow patterns from source code.
No external parser needed — works with raw C/C++ code text.
"""

import re
import numpy as np
from joblib import Parallel, delayed
from utils.llvm_compiler import extract_true_llvm_ir

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

# ================= PSEUDO-IR OPCODE EXTRACTORS =================
# Since clang cannot be guaranteed available on all host machines, we extract
# LLVM IR-like pseudo-opcodes from the source statically.

_IR_STORE_RE = re.compile(r'(?<![=!<>])=(?![=])')      # assignments
_IR_CMP_RE = re.compile(r'(==|!=|<=|>=|<|>)')           # icmp / fcmp
_IR_MATH_RE = re.compile(r'([+\-*/%])')                 # add, sub, mul, sdiv, srem
_IR_BITWISE_RE = re.compile(r'(&&|\|\||!|&|\||\^|~|<<|>>)') # and, or, xor, shl, ashr
_IR_ALLOC_RE = re.compile(r'\b(malloc|calloc|new|realloc|free|delete)\b') # allocations
_IR_VAR_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')  # generic variables (for load estimation)


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
    'loop_count', 'branch_count', 'func_call_count',
    'loop_call_combined',
    'nesting_depth', 'operator_count',
    # Algorithmic behavior features (TYPE-4 bridge)
    'return_count', 'accumulator_pattern', 'param_count', 'math_op_set_size',
    # pseudo/true IR features
    'ir_load_count', 'ir_store_count', 'ir_cmp_count',
    'ir_math_count', 'ir_bitwise_count', 'ir_br_count', 'ir_alloc_count',
    # True exclusive SSA/DataFlow IR features
    'ir_phi_count', 'ir_gep_count', 'ir_call_count'
]


def _extract_single(code):
    """Extract features for a single code sample."""
    clean_code = _strip_comments(code)
    
    ast_feats = [
        _count_loops(clean_code),
        _count_branches(clean_code),
        _count_func_calls(clean_code),
        _count_loops(clean_code) + _count_func_calls(clean_code), # loop_call_combined
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
        
    feats = ast_feats + ir_feats
    
    cf = _extract_cf_pattern(clean_code)
    return feats, cf


def extract_all_features(raw_codes):
    """
    Extract per-code numeric features and control flow patterns.
    Parallelized with joblib for speed.

    Returns:
        features: np.ndarray of shape (n_codes, num_features)
        cf_patterns: list of control flow pattern strings
    """
    print("Extracting code features (parallel)...")
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_extract_single)(code) for code in raw_codes
    )

    features = np.array([r[0] for r in results], dtype=np.float32)
    cf_patterns = [r[1] for r in results]

    return features, cf_patterns
