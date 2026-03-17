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
    # Common C identifiers in POJ-104
    "main", "printf", "scanf",
    # Preprocessor
    "include", "define",
    # Null
    "NULL", "nullptr"
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
    for tok in tokens:
        if tok in KEYWORDS:
            normalized.append(tok)
        elif _NUM_RE.match(tok):
            normalized.append("NUM")
        elif tok in LOOP_VARS:
            normalized.append(tok)
        elif len(tok) == 1 and tok.isalpha():
            normalized.append("VAR")
        elif _IDENT_RE.match(tok):
            normalized.append("VAR")
        else:
            normalized.append(tok)
    return normalized
