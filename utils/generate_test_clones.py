import os
import random
import shutil

# --- Seed Problems (Base Implementations) ---
SEEDS = {
    "fibonacci": "int fib(int n) {\n    if (n <= 1) return n;\n    int a = 0, b = 1, c;\n    for (int i = 2; i <= n; i++) {\n        c = a + b;\n        a = b;\n        b = c;\n    }\n    return b;\n}",
    "factorial": "int factorial(int n) {\n    int result = 1;\n    for (int i = 1; i <= n; i++) {\n        result *= i;\n    }\n    return result;\n}",
    "bubble_sort": "void bubbleSort(int arr[], int n) {\n    for (int i = 0; i < n-1; i++) { \n        for (int j = 0; j < n-i-1; j++) { \n            if (arr[j] > arr[j+1]) {\n                int temp = arr[j];\n                arr[j] = arr[j+1];\n                arr[j+1] = temp;\n            }\n        }\n    }\n}",
    "binary_search": "int binarySearch(int arr[], int l, int r, int x) {\n    while (l <= r) {\n        int m = l + (r - l) / 2;\n        if (arr[m] == x) return m;\n        if (arr[m] < x) l = m + 1;\n        else r = m - 1;\n    }\n    return -1;\n}",
    "gcd": "int gcd(int a, int b) {\n    while (b != 0) {\n        int temp = b;\n        b = a % b;\n        a = temp;\n    }\n    return a;\n}",
    "linear_search": "int search(int arr[], int n, int x) {\n    for (int i = 0; i < n; i++)\n        if (arr[i] == x)\n            return i;\n    return -1;\n}",
    "selection_sort": "void selectionSort(int arr[], int n) {\n    for (int i = 0; i < n-1; i++) {\n        int min_idx = i;\n        for (int j = i+1; j < n; j++)\n            if (arr[j] < arr[min_idx])\n                min_idx = j;\n        int temp = arr[min_idx];\n        arr[min_idx] = arr[i];\n        arr[i] = temp;\n    }\n}",
    "insertion_sort": "void insertionSort(int arr[], int n) {\n    for (int i = 1; i < n; i++) {\n        int key = arr[i];\n        int j = i - 1;\n        while (j >= 0 && arr[j] > key) {\n            arr[j + 1] = arr[j];\n            j = j - 1;\n        }\n        arr[j + 1] = key;\n    }\n}",
    "is_prime": "bool isPrime(int n) {\n    if (n <= 1) return false;\n    for (int i = 2; i * i <= n; i++)\n        if (n % i == 0)\n            return false;\n    return true;\n}",
    "power": "int power(int base, int exp) {\n    int res = 1;\n    for (int i = 0; i < exp; i++)\n        res *= base;\n    return res;\n}",
    "reverse_string": "void reverseString(char* str) {\n    int l = 0, r = strlen(str) - 1;\n    while (l < r) {\n        char temp = str[l];\n        str[l] = str[r];\n        str[r] = temp;\n        l++; r--;\n    }\n}",
    "palindrome_check": "bool isPalindrome(char* str) {\n    int l = 0, r = strlen(str) - 1;\n    while (l < r) {\n        if (str[l] != str[r]) return false;\n        l++; r--;\n    }\n    return true;\n}",
    "find_max": "int findMax(int arr[], int n) {\n    int max = arr[0];\n    for (int i = 1; i < n; i++)\n        if (arr[i] > max)\n            max = arr[i];\n    return max;\n}",
    "find_min": "int findMin(int arr[], int n) {\n    int min = arr[0];\n    for (int i = 1; i < n; i++)\n        if (arr[i] < min)\n            min = arr[i];\n    return min;\n}",
    "array_sum": "int sum(int arr[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++)\n        total += arr[i];\n    return total;\n}",
    "count_vowels": "int countVowels(char* str) {\n    int count = 0;\n    for (int i = 0; str[i] != '\\0'; i++) {\n        char c = tolower(str[i]);\n        if (c=='a' || c=='e' || c=='i' || c=='o' || c=='u')\n            count++;\n    }\n    return count;\n}",
    "lcm": "int lcm(int a, int b) {\n    int max = (a > b) ? a : b;\n    while (1) {\n        if (max % a == 0 && max % b == 0)\n            return max;\n        max++;\n    }\n}",
    "string_length": "int stringLength(char* str) {\n    int len = 0;\n    while (str[len] != '\\0')\n        len++;\n    return len;\n}",
    "swap_numbers": "void swap(int *xp, int *yp) {\n    int temp = *xp;\n    *xp = *yp;\n    *yp = temp;\n}",
    "sum_of_digits": "int sumOfDigits(int n) {\n    int sum = 0;\n    while (n != 0) {\n        sum += n % 10;\n        n /= 10;\n    }\n    return sum;\n}"
}

# --- Type 4 Pairs (Semantic Clones: Different Implementation, Same Logic) ---
TYPE_4_PAIRS = [
    # Fibonacci: Iterative vs Recursive
    ("int fib(int n) {\n    if (n <= 1) return n;\n    int a = 0, b = 1, c;\n    for (int i = 2; i <= n; i++) {\n        c = a + b;\n        a = b;\n        b = c;\n    }\n    return b;\n}",
     "int fib_recursive(int n) {\n    if (n <= 1) return n;\n    return fib_recursive(n-1) + fib_recursive(n-2);\n}"),
    # Factorial: Iterative vs Recursive
    ("int factorial(int n) {\n    int result = 1;\n    for (int i = 1; i <= n; i++) {\n        result *= i;\n    }\n    return result;\n}",
     "int factorial_recursive(int n) {\n    if (n == 0 || n == 1) return 1;\n    return n * factorial_recursive(n-1);\n}"),
    # Binary Search: Iterative vs Recursive
    ("int binarySearch(int arr[], int l, int r, int x) {\n    while (l <= r) {\n        int m = l + (r - l) / 2;\n        if (arr[m] == x) return m;\n        if (arr[m] < x) l = m + 1;\n        else r = m - 1;\n    }\n    return -1;\n}",
     "int binarySearchRec(int arr[], int l, int r, int x) {\n    if (r >= l) {\n        int mid = l + (r - l) / 2;\n        if (arr[mid] == x) return mid;\n        if (arr[mid] > x) return binarySearchRec(arr, l, mid - 1, x);\n        return binarySearchRec(arr, mid + 1, r, x);\n    }\n    return -1;\n}"),
    # GCD: Iterative Euclidean vs Recursive Euclidean
    ("int gcd(int a, int b) {\n    while (b != 0) {\n        int temp = b;\n        b = a % b;\n        a = temp;\n    }\n    return a;\n}",
     "int gcd_recursive(int a, int b) {\n    if (b == 0) return a;\n    return gcd_recursive(b, a % b);\n}"),
    # Is Prime: Simple Loop vs 6k+-1 optimization
    ("bool isPrime(int n) {\n    if (n <= 1) return false;\n    for (int i = 2; i * i <= n; i++)\n        if (n % i == 0)\n            return false;\n    return true;\n}",
     "bool isPrimeOpt(int n) {\n    if (n <= 1) return false;\n    if (n <= 3) return true;\n    if (n % 2 == 0 || n % 3 == 0) return false;\n    for (int i = 5; i * i <= n; i = i + 6)\n        if (n % i == 0 || n % (i + 2) == 0)\n            return false;\n    return true;\n}"),
    # Array Sum: Iterative vs Recursive
    ("int sum(int arr[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++)\n        total += arr[i];\n    return total;\n}",
     "int sumRec(int arr[], int n) {\n    if (n <= 0) return 0;\n    return (sumRec(arr, n - 1) + arr[n - 1]);\n}"),
    # String Length: While loop vs For loop
    ("int stringLength(char* str) {\n    int len = 0;\n    while (str[len] != '\\0')\n        len++;\n    return len;\n}",
     "int stringLengthFor(char* str) {\n    int len = 0;\n    for (; str[len]; len++);\n    return len;\n}"),
    # Swap Numbers: Temp variable vs XOR
    ("void swap(int *xp, int *yp) {\n    int temp = *xp;\n    *xp = *yp;\n    *yp = temp;\n}",
     "void swapXor(int *xp, int *yp) {\n    if (xp != yp) {\n        *xp = *xp ^ *yp;\n        *yp = *xp ^ *yp;\n        *xp = *xp ^ *yp;\n    }\n}"),
    # Power: Iterative vs Recursive (Divide and Conquer)
    ("int power(int base, int exp) {\n    int res = 1;\n    for (int i = 0; i < exp; i++)\n        res *= base;\n    return res;\n}",
     "int powerFast(int base, int exp) {\n    if (exp == 0) return 1;\n    int half = powerFast(base, exp / 2);\n    if (exp % 2 == 0) return half * half;\n    else return base * half * half;\n}"),
    # Find Max: For loop vs Recursive
    ("int findMax(int arr[], int n) {\n    int max = arr[0];\n    for (int i = 1; i < n; i++)\n        if (arr[i] > max)\n            max = arr[i];\n    return max;\n}",
     "int findMaxRec(int arr[], int n) {\n    if (n == 1) return arr[0];\n    int m = findMaxRec(arr, n - 1);\n    return (arr[n-1] > m) ? arr[n-1] : m;\n}"),
    # Count Vowels: for loop vs while loop
    ("int countVowels(char* str) {\n    int count = 0;\n    for (int i = 0; str[i] != '\\0'; i++) {\n        char c = tolower(str[i]);\n        if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u') count++;\n    }\n    return count;\n}",
     "int countVowelsW(char* s) {\n    int cnt = 0, i = 0;\n    while (s[i]) {\n        char c = tolower(s[i++]);\n        if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u') cnt++;\n    }\n    return cnt;\n}"),
    # Is Palindrome: two-pointer vs recursive
    ("bool isPalindrome(char* str) {\n    int l = 0, r = strlen(str) - 1;\n    while (l < r) {\n        if (str[l] != str[r]) return false;\n        l++; r--;\n    }\n    return true;\n}",
     "bool isPalRec(char* s, int l, int r) {\n    if (l >= r) return true;\n    if (s[l] != s[r]) return false;\n    return isPalRec(s, l+1, r-1);\n}"),
    # Sum of digits: while vs recursive
    ("int sumOfDigits(int n) {\n    int sum = 0;\n    while (n != 0) { sum += n % 10; n /= 10; }\n    return sum;\n}",
     "int sumDigRec(int n) {\n    if (n == 0) return 0;\n    return n % 10 + sumDigRec(n / 10);\n}"),
    # LCM: brute force vs GCD-based
    ("int lcm(int a, int b) {\n    int max = (a > b) ? a : b;\n    while (1) { if (max % a == 0 && max % b == 0) return max; max++; }\n}",
     "int gcdH(int a, int b) { return b == 0 ? a : gcdH(b, a % b); }\n"
     "int lcmFast(int a, int b) { return (a / gcdH(a, b)) * b; }"),
]

# We need 100 type 4 pairs, so we will generate variants of these base pairs by adding whitespace, comments, etc., but they will remain semantically identical but structurally different.

def apply_type1(code):
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        if random.random() < 0.2:
            new_lines.append(f"// {random.choice(['TODO', 'Fix this', 'Check if needed', 'Logic here'])}")
        new_lines.append((" " * random.randint(0, 4)) + line + (" " * random.randint(0, 4)))
        if random.random() < 0.1:
            new_lines.append("")
    return '\n'.join(new_lines)

def apply_type2(code: str) -> str:
    """Simple token renaming for Type-2 clone generation."""
    import re
    replacements = {
        "num_elements": "n",  # reverse order to avoid double-replace
        "n": "num_elements",
        "i": "idx",
        "j": "idx2",
        "a": "val1",
        "b": "val2",
        "temp": "tmp_var",
        "result": "res",
        "arr": "array_data",
        "max": "maximum_val",
        "min": "minimum_val",
        "total": "total_sum",
        "count": "counter",
        "l": "left",
        "r": "right",
        "m": "mid",
        "x": "target",
    }
    # C/C++ keywords that must never be replaced
    _SKIP_KEYWORDS = frozenset({
        "int", "float", "double", "char", "void", "long", "short",
        "return", "if", "else", "for", "while", "do", "break",
        "continue", "switch", "case", "struct", "const", "bool",
    })
    for old, new in replacements.items():
        if old in _SKIP_KEYWORDS or new in _SKIP_KEYWORDS:
            continue
        # Word-boundary replacement, but only outside string literals
        code = re.sub(rf'(?<![a-zA-Z_])\b{re.escape(old)}\b(?![a-zA-Z_0-9])', new, code)
    return code

def apply_type3(code):
    # near-miss modifications
    code = apply_type2(code)
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        # statement reordering or trivial addition
        if "{" in line and random.random() < 0.3:
            new_lines.append(line)
            new_lines.append(f"    int dummy_{random.randint(0,100)} = 0;")
        elif "return" in line and random.random() < 0.5:
            new_lines.append(f"    // returning result")
            new_lines.append(line)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def generate_pairs():
    out_dir = "test_clones"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    types = ["type1", "type2", "type3", "type4", "negatives"]
    for t in types:
        os.makedirs(os.path.join(out_dir, t))
        
    all_snippets = [] # Store tuples of (seed_name, code) to generate negatives later
        
    # Generate Type 1, 2, 3
    seed_keys = list(SEEDS.keys())
    
    for t_idx, t_name in enumerate(["type1", "type2", "type3"]):
        pair_idx = 1
        for seed_k in seed_keys:
            base_code = SEEDS[seed_k]
            # Generate 5-6 variations per seed to get >100
            for _ in range(6):
                if t_name == "type1":
                    clone_code = apply_type1(base_code)
                elif t_name == "type2":
                    clone_code = apply_type2(base_code)
                elif t_name == "type3":
                    clone_code = apply_type3(base_code)
                
                pair_dir = os.path.join(out_dir, t_name, f"pair_{pair_idx:03d}")
                os.makedirs(pair_dir)
                with open(os.path.join(pair_dir, "original.txt"), "w") as f:
                    f.write(base_code)
                with open(os.path.join(pair_dir, "clone.txt"), "w") as f:
                    f.write(clone_code)
                all_snippets.append((seed_k, base_code))
                all_snippets.append((seed_k, clone_code))
                pair_idx += 1

    # Generate Type 4
    pair_idx = 1
    # Her base pair üzerinde 11 varyasyon oluştur (0.15 olasılıkla orijinal kod kullan)
    for base, alt in TYPE_4_PAIRS:
        for _ in range(11):
            # 0.15 olasılıkla orijinal, 0.85 olasılıkla type1/2 varyasyon uy. (#16 düz.)
            c1 = base if random.random() < 0.15 else apply_type1(apply_type2(base))
            c2 = alt  if random.random() < 0.15 else apply_type1(apply_type2(alt))
            
            pair_dir = os.path.join(out_dir, "type4", f"pair_{pair_idx:03d}")
            os.makedirs(pair_dir)
            with open(os.path.join(pair_dir, "original.txt"), "w") as f:
                f.write(c1)
            with open(os.path.join(pair_dir, "clone.txt"), "w") as f:
                f.write(c2)
            
            # Since TYPE_4_PAIRS are in order of SEEDS (fib, fact, binarySearch...) we can roughly assign names or just use a generic 't4_seed'
            all_snippets.append((f"t4_{pair_idx}", c1))
            all_snippets.append((f"t4_{pair_idx}", c2))
            pair_idx += 1

    # Generate 8930 Negatives to reach 5% positive ratio (470 positives / 0.05 = 9400 total)
    print("Generating 8930 negative pairs for realistic 0.05 positive ratio...")
    pair_idx = 1
    negative_pairs = set()
    max_attempts = 100000
    attempts = 0
    
    while pair_idx <= 8930 and attempts < max_attempts:
        s1 = random.choice(all_snippets)
        s2 = random.choice(all_snippets)
        
        # Check if they belong to different seed algorithms
        if s1[0] != s2[0]:
            # Tuple hash: hash((s1,s2)) ≠ hash((s2,s1)) ve
            # "AB"+"C" == "A"+"BC" string concaténasyon çakışmasını önler.
            pair_hash = hash((s1[1], s2[1]))
            if pair_hash not in negative_pairs:
                negative_pairs.add(pair_hash)
                
                pair_dir = os.path.join(out_dir, "negatives", f"pair_{pair_idx:04d}")
                os.makedirs(pair_dir)
                with open(os.path.join(pair_dir, "original.txt"), "w") as f:
                    f.write(s1[1])
                with open(os.path.join(pair_dir, "clone.txt"), "w") as f:
                    f.write(s2[1])
                pair_idx += 1
        attempts += 1

    print(f"Dataset generated successfully in {out_dir}/")

if __name__ == "__main__":
    generate_pairs()
