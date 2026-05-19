"""
C/C++ Tokenizer & Normalizer — preprocessing/tokenizer.py
==========================================================
Tokenizes C/C++ source code and normalizes tokens for ML feature extraction.

Normalization rules (order matters):
  - Comments                   → stripped before tokenization (not tokens)
  - String/char literals       → STR_LIT      (handles C++11 Raw Strings)
  - Preprocessor directives    → PREPROC       (single token, #include, #define, ...)
  - All numeric literals        → NUM           (int, float, hex, binary, hex-float + suffixes)
  - C/C++ keywords              → kept as-is    (for, while, int, ...)
  - C++ Stdlib Functions        → masked to <IO_FUNC>, <MEM_FUNC>, <MATH_FUNC> etc.
  - User-defined identifiers:
      followed by '('          → FUNC
      otherwise                → VAR
  - Operators/punctuation       → kept as-is    (handles <=> spaceship operator)

Design decisions:
  - C++ Stdlib Masking: `printf` and `cout` both map to `IO_FUNC`. This massively
    improves Type-3 detection because differing IO/Mem mechanisms are normalized.
  - Whitespace is ignored entirely (C++ semantics).
"""

import re
from utils.logger import Log


# -- C/C++ Core Keywords -------------------------------------------------------
KEYWORDS = frozenset({
    # Primitive types
    "int", "float", "double", "char", "void", "long", "short",
    "unsigned", "signed", "bool", "wchar_t", "size_t", "auto",
    # Control flow
    "return", "if", "else", "for", "while", "do", "break",
    "continue", "switch", "case", "default", "goto",
    # Compound types
    "struct", "typedef", "enum", "union", "sizeof", "class",
    # Storage/qualifiers
    "static", "const", "extern", "volatile", "register", "inline",
    # Boolean / Null
    "true", "false", "NULL", "nullptr",
    # C++ OOP / Modern
    "public", "private", "protected", "virtual", "template", "typename",
    "namespace", "using", "this", "override", "operator", "explicit",
    "friend", "mutable", "constexpr", "consteval", "constinit",
    # Special runtime kept verbatim
    "main", "std"
})

# -- STDLIB Masking Groups -----------------------------------------------------
STDLIB_IO = {"printf", "scanf", "cout", "cin", "cerr", "endl", "getline", "fopen", "fclose", "fread", "fwrite", "fprintf", "fscanf", "puts", "gets"}
STDLIB_MEM = {"malloc", "calloc", "realloc", "free", "new", "delete", "memset", "memcpy", "memmove", "memcmp"}
STDLIB_MATH = {"abs", "fabs", "pow", "sqrt", "sin", "cos", "tan", "log", "log10", "exp", "ceil", "floor", "round", "max", "min", "swap"}
STDLIB_STR = {"strlen", "strcpy", "strncpy", "strcmp", "strncmp", "strcat", "strchr", "strstr", "tolower", "toupper", "isalpha", "isdigit"}
STDLIB_DATA = {"vector", "string", "map", "set", "list", "stack", "queue", "deque", "priority_queue", "unordered_map", "unordered_set", "pair", "tuple", "make_pair", "make_tuple", "sort"}

# Flatten dictionary for O(1) lookup
STDLIB_LOOKUP = {}
for group_name, group_set in [
    ("IO_FUNC", STDLIB_IO),
    ("MEM_FUNC", STDLIB_MEM),
    ("MATH_FUNC", STDLIB_MATH),
    ("STR_FUNC", STDLIB_STR),
    ("DATA_STRUCT", STDLIB_DATA),
]:
    for func in group_set:
        STDLIB_LOOKUP[func] = group_name

# -- Master tokenizer regex ----------------------------------------------------
# IMPORTANT: alternation order determines priority — more specific / longer
# patterns MUST come before shorter ones.
_TOKEN_RE = re.compile(
    # 1. Raw string literals (C++11): R"prefix(content)prefix"
    r'R"([a-zA-Z0-9_]*)\(.*?\)\1"'
    # 2. String/char literals (handles escapes)
    r'|"(?:[^"\\]|\\.)*"'              # "hello\n", ""
    r"|'(?:[^'\\]|\\.)*'"             # 'x', '\n', '\\'
    # 3. Preprocessor directives
    r"|#[^\n]*"                       # #include <stdio.h>, #define MAX 100
    # 4. Numeric literals (Hex-floats, binary, hex, decimal with optional suffixes)
    r"|0[xX][0-9a-fA-F]+\.[0-9a-fA-F]*[pP][+-]?\d+[fFlL]?" # hex-float 0x1.p3
    r"|0[xX][0-9a-fA-F]+[uUlLfF]{0,3}"  # hex: 0xFF, 0xDEADUL
    r"|0[bB][01]+[uUlLfF]{0,3}"          # binary: 0b1010
    r"|\d+\.\d*(?:[eE][+-]?\d+)?[fFlL]?" # float: 3.14, 3.14e-10, 3.14f
    r"|\.\d+(?:[eE][+-]?\d+)?[fFlL]?"    # float: .14, .14e5f
    r"|\d+[uUlLfF]{0,3}"                 # integer: 42, 42UL, 1000LL
    # 5. Identifiers
    r"|[A-Za-z_]\w*"
    # 6. Multi-char C/C++ operators
    r"|<=>|\.\.\.|->\*|\.\*"         # spaceship <=>, variadic ..., pointer-to-member
    r"|::|<<=|>>="                   # scope resolution, compound shift-assign
    r"|&&|\|\||<<|>>"                # logical, bitshift
    r"|\+\+|--|->|"                  # increment/decrement, arrow
    r"==|!=|<=|>="                   # comparison
    r"|[+\-*/%&|^]="                 # compound assign
    # 7. Single-char operators and punctuation
    r"|[%+\-*/=<>;(){}[\],&|!~^?:.]",
    re.DOTALL,
)

# -- Per-token classification regexes ------------------------------------------

# Comment stripping — block BEFORE line (block comments can span multiple lines)
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_LINE_COMMENT_RE  = re.compile(r'//[^\n]*')

# Numeric suffix validation
_NUM_SUFFIX_RE = re.compile(
    r"^("
    r"0[xX][0-9a-fA-F]+\.[0-9a-fA-F]*[pP][+-]?\d+" # hex-float
    r"|0[xX][0-9a-fA-F]+"     # hex body
    r"|0[bB][01]+"            # binary body
    r"|\d+\.\d*(?:[eE][+-]?\d+)?" # float
    r"|\.\d+(?:[eE][+-]?\d+)?"    # float
    r"|\d+"                   # integer
    r")[uUlLfF]{0,3}$"        # optional C/C++ suffix
)

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z_0-9]*$")


def _strip_comments(code: str) -> str:
    """Remove C/C++ block and line comments from source before tokenization."""
    code = _BLOCK_COMMENT_RE.sub(' ', code)  # space preserves column positions
    code = _LINE_COMMENT_RE.sub('', code)
    return code


def tokenize(code: str) -> list:
    """Return raw tokens from C/C++ source code string."""
    return [m.group(0) for m in _TOKEN_RE.finditer(_strip_comments(code)) if m.group(0).strip()]


def normalize_tokens(tokens: list) -> list:
    """
    Map raw tokens to normalized form for TF-IDF / cosine similarity.
    Includes STDLIB Masking for vastly improved Type-3 matching.
    """
    normalized = []
    length = len(tokens)
    for idx in range(length):
        tok = tokens[idx]

        # 1. Raw string literals
        if tok.startswith('R"') or (tok.startswith('"') or tok.startswith("'")):
            normalized.append("STR_LIT")

        # 2. Preprocessor directive
        elif tok.startswith("#"):
            normalized.append("PREPROC")

        # 3. Keyword -- preserved verbatim
        elif tok in KEYWORDS:
            normalized.append(tok)

        # 4. Stdlib Masking (printf -> IO_FUNC)
        elif tok in STDLIB_LOOKUP:
            normalized.append(STDLIB_LOOKUP[tok])

        # 5. Numeric literal
        elif _NUM_SUFFIX_RE.match(tok):
            normalized.append("NUM")

        # 6. Identifier -- function or variable
        elif _IDENT_RE.match(tok):
            # Lookahead: identifier immediately followed by '(' -> function call/def
            if idx + 1 < length and tokens[idx + 1] == "(":
                normalized.append("FUNC")
            else:
                normalized.append("VAR")

        # 7. Operators, punctuation -- preserved verbatim
        else:
            normalized.append(tok)

    return normalized

if __name__ == "__main__":
    # Test script built-in
    test_code = '''
    #include <iostream>
    using namespace std;
    int main() {
        // Init vector
        vector<int> arr;
        int max_val = 0xFFUL;
        float x = 0x1.p3; // hex float
        auto msg = R"DELIM(this is a (raw) string)DELIM";
        
        for (int i=0; i<10; i++) {
            arr.push_back(i);
            if (i <=> 5 > 0) {
                printf("%d", i);
            } else {
                cout << i;
            }
        }
        return 0;
    }
    '''
    print("RAW TOKENS:")
    raw = tokenize(test_code)
    print(raw)
    print("\nNORMALIZED:")
    print(normalize_tokens(raw))
