import re

KEYWORDS = {
    "int","float","double","char","void","return","if","else","for",
    "while","break","continue","switch","case","default","struct",
    "static","const","true","false"
}

LOOP_VARS = {"i", "j", "k", "n"}

def tokenize(code):
    return re.findall(
        r"[A-Za-z_]+|\d+|==|<=|>=|!=|[%+\-*/=<>;(){}]",
        code
    )

def normalize_tokens(tokens):
    normalized = []
    for tok in tokens:
        if tok in KEYWORDS:
            normalized.append(tok)
        elif tok.isdigit():
            normalized.append("NUM")
        elif tok in LOOP_VARS:
            normalized.append(tok)
        elif len(tok) == 1 and tok.isalpha():
            normalized.append("VAR")
        elif re.match(r"[a-zA-Z_][a-zA-Z_0-9]*", tok):
            normalized.append("VAR")
        else:
            normalized.append(tok)
    return normalized
