from preprocessing.code_features import _extract_single

code1 = """
int main() {
    for(int i=0; i<10; i++) {
        if(i > 5) {
            printf("%d", i);
        }
    }
}
"""

code2 = """
int main() {
    for(int i=0; i<10; i++)
        if(i > 5)
            printf("%d", i);
}
"""

code3 = """
int main() {
    string s = "{ this { is } a test { }"; // SHOULD BE IGNORED
    return 0;
}
"""

f1, _, _ = _extract_single(code1)
f2, _, _ = _extract_single(code2)
f3, _, _ = _extract_single(code3)

print("Depth 1 (with braces):", f1[2])
print("Depth 2 (without braces):", f2[2])
print("Depth 3 (braces inside string):", f3[2])
