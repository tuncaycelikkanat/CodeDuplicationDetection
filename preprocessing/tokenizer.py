import re

KEYWORDS = {
    # Types
    "int", "float", "double", "char", "void", "long", "short",
    "unsigned", "signed", "bool",
    # Control flow
    "return", "if", "else", "for", "while", "do", "break",
    "continue", "switch", "case", "default", "goto",
    # Structures
    "struct", "typedef", "enum", "union", "sizeof",
    # Qualifiers
    "static", "const", "extern", "volatile",
    # Boolean
    "true", "false",
    # Common C/C++ identifiers in POJ-104
    "main", "printf", "scanf", "cin", "cout", "endl", "std",
    # Preprocessor
    "include", "define",
    # Null and Memory
    "NULL", "nullptr", "malloc", "calloc", "realloc", "free", "new", "delete"
}

LOOP_VARS = {"i", "j", "k", "n", "m", "l", "x", "y"}

# Pre-compiled regex patterns for performance
_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*"                  # identifiers
    r"|\d+\.?\d*"                     # numbers (including floats)
    r"|&&|\|\||<<|>>|\+\+|--|->"      # multi-char operators
    r"|==|<=|>=|!="                   # comparison operators
    r"|[%+\-*/=<>;(){}[\],&|!~^?:#]"  # single-char operators and punctuation
)

_NUM_RE = re.compile(r"^\d+\.?\d*$")
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z_0-9]*$")


def tokenize(code):
    return _TOKEN_RE.findall(code)


def normalize_tokens(tokens):
    normalized = []
    length = len(tokens)
    for i in range(length):
        tok = tokens[i]
        if tok in KEYWORDS:
            normalized.append(tok)
        elif _NUM_RE.match(tok):
            normalized.append("NUM")
        elif tok in LOOP_VARS:
            normalized.append(tok)
        elif _IDENT_RE.match(tok):
            # Lookahead: if followed by '(', it's a function call or definition
            if i + 1 < length and tokens[i+1] == "(":
                normalized.append("FUNC")
            else:
                # Tek karakterli ya da çok karakterli tüm tanımlayıcılar VAR'a dönüşür
                normalized.append("VAR")
        else:
            normalized.append(tok)
    return normalized
