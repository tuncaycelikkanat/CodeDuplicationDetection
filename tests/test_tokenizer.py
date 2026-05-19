"""
Kapsamlı unit testler: preprocessing/tokenizer.py
==================================================
Testler 8 gruba ayrılmıştır:
  A) Temel tokenizasyon (tokenize)
  B) Sayısal literaller
  C) String / Char literaller
  D) Preprocessor direktifleri
  E) Operatörler (multi-char, C++ özel)
  F) Normalizasyon (normalize_tokens)
  G) Tutarlılık & Geriye dönük uyumluluk
  H) Type-2 Klon Tespiti için Kritik Testler
"""
import sys, os
from utils.logger import Log

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.tokenizer import tokenize, normalize_tokens, KEYWORDS


# ─────────────────────────────────────────────────────────────────────────────
# A) Temel Tokenizasyon
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizeBasic:
    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_simple_declaration(self):
        tokens = tokenize("int x = 5;")
        assert "int" in tokens
        assert "x" in tokens
        assert "5" in tokens
        assert "=" in tokens
        assert ";" in tokens

    def test_function_definition(self):
        tokens = tokenize("int fib(int n) {")
        assert "int" in tokens
        assert "fib" in tokens
        assert "(" in tokens
        assert "n" in tokens
        assert "{" in tokens

    def test_whitespace_and_newlines_ignored(self):
        t1 = tokenize("int   x = 5;")
        t2 = tokenize("int x = 5;")
        assert t1 == t2

    def test_comment_block_ignored(self):
        """Block comment içeriği token'a dönüşmemeli (regex içeriği yakalamalı)"""
        # /* ... */ yorumları _TOKEN_RE tarafından yakalanmaz — içleri gürültü olur.
        # Bu test mevcut davranışı belgeliyor: gelecekte fix edilebilir.
        tokens = tokenize("int x = /* comment */ 5;")
        assert "int" in tokens
        assert "x" in tokens
        assert "5" in tokens
        # 'comment' identifier olarak yakalanıyor — bilinen sınırlama
        # (Bu test geçerliyse yorum soyutlama EKLENMEDİ demektir)


# ─────────────────────────────────────────────────────────────────────────────
# B) Sayısal Literaller — Tokenize Aşaması
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizeNumbers:
    def test_integer(self):
        assert "42" in tokenize("42")

    def test_zero(self):
        assert "0" in tokenize("0")

    def test_float_standard(self):
        """3.14 tek token olarak yakalanmalı"""
        tokens = tokenize("3.14")
        assert "3.14" in tokens
        assert "3" not in tokens  # İki ayrı token olmamalı
        assert ".14" not in tokens

    def test_float_trailing_dot(self):
        """3. (trailing dot float) tek token olmalı"""
        tokens = tokenize("3.")
        # 3. → float literal olarak tek token
        assert "3." in tokens

    def test_float_leading_dot(self):
        """.14 (leading dot float) tek token olmalı"""
        tokens = tokenize(".14")
        assert ".14" in tokens
        assert "14" not in tokens  # Ayrı token olmamalı

    def test_hex_number(self):
        """0xFF → tek token, identifier parçasına ayrılmamalı"""
        tokens = tokenize("0xFF")
        assert "0xFF" in tokens
        assert "xFF" not in tokens   # Identifier olarak ayrılmamalı
        assert len([t for t in tokens if t == "0xFF"]) == 1

    def test_hex_uppercase(self):
        tokens = tokenize("0XDEADBEEF")
        assert "0XDEADBEEF" in tokens

    def test_binary_number(self):
        """0b1010 → tek token (GCC extension)"""
        tokens = tokenize("0b1010")
        assert "0b1010" in tokens
        assert "b1010" not in tokens  # Identifier olarak ayrılmamalı

    def test_binary_uppercase_B(self):
        tokens = tokenize("0B1111")
        assert "0B1111" in tokens

    def test_hex_in_expression(self):
        tokens = tokenize("x = 0xFF + 42;")
        assert "0xFF" in tokens
        assert "42" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# C) String ve Char Literaller
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizeStringLiterals:
    def test_string_literal_single_token(self):
        """\"hello\" → tek token olmalı, içerik ayrı ayrı tokenlanmamalı"""
        tokens = tokenize('"hello"')
        assert '"hello"' in tokens
        assert "h" not in tokens
        assert "e" not in tokens
        assert "hello" not in tokens  # İçerik identifier olarak çıkmamalı

    def test_string_with_spaces(self):
        tokens = tokenize('"hello world"')
        assert '"hello world"' in tokens
        # "world" ayrı identifier olarak çıkmamalı
        assert "world" not in tokens

    def test_string_with_escape(self):
        """Escape sequence içeren string literal tek token olmalı"""
        tokens = tokenize(r'"hello\nworld"')
        assert len([t for t in tokens if t.startswith('"')]) == 1

    def test_printf_format_string(self):
        """printf("Result: %d\n", x) — format string tek token"""
        tokens = tokenize('printf("Result: %d\\n", x)')
        # Format string content'i ayrı token olmamalı
        assert "Result" not in tokens
        # printf, (, STR_LIT, ',', x, ) olmalı
        assert "printf" in tokens
        assert "x" in tokens

    def test_char_literal(self):
        """'x' → tek token"""
        tokens = tokenize("'x'")
        assert "'x'" in tokens

    def test_char_escape_literal(self):
        r"""'\n' → tek token"""
        tokens = tokenize(r"'\n'")
        assert len([t for t in tokens if t.startswith("'")]) == 1

    def test_empty_string_literal(self):
        tokens = tokenize('""')
        assert '""' in tokens


# ─────────────────────────────────────────────────────────────────────────────
# D) Preprocessor Direktifleri
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizePreprocessor:
    def test_include_is_single_token(self):
        """#include <stdio.h> → tek token, stdio ayrı identifier olmamalı"""
        tokens = tokenize("#include <stdio.h>")
        # Tüm satır tek token olarak yakalanmalı
        assert any(t.startswith("#") for t in tokens)
        assert len([t for t in tokens if t.startswith("#")]) == 1
        # stdio identifier olarak çıkmamalı
        assert "stdio" not in tokens

    def test_define_single_token(self):
        tokens = tokenize("#define MAX 100")
        assert any(t.startswith("#") for t in tokens)
        assert "MAX" not in tokens  # İçeriği ayrı token olmamalı

    def test_ifdef(self):
        tokens = tokenize("#ifdef DEBUG")
        assert any(t.startswith("#") for t in tokens)

    def test_pragma(self):
        tokens = tokenize("#pragma once")
        assert any(t.startswith("#") for t in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# E) Operatörler
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizeOperators:
    def test_standard_multi_char_operators(self):
        tokens = tokenize("a == b && c != d || e <= f")
        assert "==" in tokens
        assert "&&" in tokens
        assert "!=" in tokens
        assert "||" in tokens
        assert "<=" in tokens

    def test_increment_decrement(self):
        tokens = tokenize("i++ j--")
        assert "++" in tokens
        assert "--" in tokens

    def test_arrow_operator(self):
        """-> pointer member access"""
        tokens = tokenize("node->next")
        assert "->" in tokens
        assert len([t for t in tokens if t == "->"]) == 1

    def test_scope_resolution(self):
        """:: C++ scope resolution operatörü tek token olmalı"""
        tokens = tokenize("std::sort")
        assert "::" in tokens
        # : : olarak ikiye bölünmemeli
        assert tokens.count(":") == 0

    def test_scope_resolution_split_check(self):
        """std::sort → ['std', '::', 'sort'] — ikiye bölünmemeli"""
        tokens = tokenize("std::sort(arr, arr+n)")
        assert "::" in tokens
        # Sadece bir tane :: token olmalı
        assert tokens.count("::") == 1

    def test_pointer_to_member_dot(self):
        """obj.*ptr → .* tek token"""
        tokens = tokenize("obj.*ptr")
        assert ".*" in tokens

    def test_pointer_to_member_arrow(self):
        """obj->*ptr → ->* tek token"""
        tokens = tokenize("ptr->*member")
        assert "->*" in tokens

    def test_variadic(self):
        """... variadic tek token"""
        tokens = tokenize("void func(...)")
        assert "..." in tokens

    def test_bitshift(self):
        tokens = tokenize("x << 2")
        assert "<<" in tokens

    def test_compound_assign(self):
        tokens = tokenize("x += 5; y -= 3;")
        assert "+=" in tokens
        assert "-=" in tokens

    def test_logical_not(self):
        tokens = tokenize("if (!flag)")
        assert "!" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# F) Normalizasyon (normalize_tokens)
# ─────────────────────────────────────────────────────────────────────────────
class TestNormalizeTokens:

    # ── F.1 Keywords ──
    def test_keyword_preserved(self):
        norm = normalize_tokens(tokenize("int x;"))
        assert "int" in norm

    def test_all_control_flow_keywords_preserved(self):
        code = "for (int i = 0; i < n; i++) { if (x) break; else continue; }"
        norm = normalize_tokens(tokenize(code))
        for kw in ("for", "if", "break", "else", "continue"):
            assert kw in norm, f"Keyword '{kw}' should be preserved"

    def test_keyword_not_func(self):
        """for( — 'for' keyword olarak kalmalı, FUNC olmamalı"""
        norm = normalize_tokens(tokenize("for(int i=0;"))
        assert "for" in norm
        assert "FUNC" not in norm

    # ── F.2 Numbers ──
    def test_integer_normalized(self):
        norm = normalize_tokens(tokenize("42"))
        assert "NUM" in norm
        assert "42" not in norm

    def test_float_normalized(self):
        norm = normalize_tokens(tokenize("3.14"))
        assert "NUM" in norm
        assert "3.14" not in norm

    def test_float_leading_dot_normalized(self):
        norm = normalize_tokens(tokenize(".5"))
        assert "NUM" in norm

    def test_hex_normalized(self):
        """0xFF → NUM"""
        norm = normalize_tokens(tokenize("0xFF"))
        assert "NUM" in norm
        assert "0xFF" not in norm
        assert "xFF" not in norm

    def test_binary_normalized(self):
        """0b1010 → NUM"""
        norm = normalize_tokens(tokenize("0b1010"))
        assert "NUM" in norm
        assert "0b1010" not in norm

    # ── F.3 Identifiers ──
    def test_user_variable_becomes_var(self):
        norm = normalize_tokens(tokenize("myVariable = 1;"))
        assert "VAR" in norm
        assert "myVariable" not in norm

    def test_function_call_becomes_func(self):
        norm = normalize_tokens(tokenize("myFunc(x)"))
        assert "FUNC" in norm
        assert "myFunc" not in norm

    def test_function_definition_becomes_func(self):
        norm = normalize_tokens(tokenize("int compute(int x)"))
        assert "FUNC" in norm

    # ── F.4 Loop Variables (kritik değişiklik) ──
    def test_loop_variable_i_becomes_var(self):
        """i döngü değişkeni artık VAR olmalı, korunmamalı"""
        norm = normalize_tokens(tokenize("for (int i = 0; i < n; i++)"))
        assert "VAR" in norm
        # 'i' artık VAR'a dönüşüyor — eski davranış (korunma) BUG'dı
        # 'i' tokenizer'dan 'i' olarak çıkıyor ama normalize → VAR
        assert "i" not in norm, (
            "'i' should normalize to VAR, not be preserved as 'i'. "
            "This ensures Type-2 clones (i→idx) produce identical token sequences."
        )

    def test_loop_variable_j_becomes_var(self):
        norm = normalize_tokens(tokenize("j < n"))
        assert "j" not in norm
        assert "VAR" in norm

    def test_loop_variable_n_becomes_var(self):
        norm = normalize_tokens(tokenize("int n = 10;"))
        # n VAR'a dönüşmeli
        assert "n" not in norm
        assert "VAR" in norm

    # ── F.5 String & Char Literaller ──
    def test_string_literal_becomes_str_lit(self):
        norm = normalize_tokens(tokenize('"hello"'))
        assert "STR_LIT" in norm
        assert "hello" not in norm

    def test_char_literal_becomes_str_lit(self):
        norm = normalize_tokens(tokenize("'x'"))
        assert "STR_LIT" in norm

    def test_printf_format_string_normalized(self):
        """printf("fmt", x) → format string STR_LIT olmalı"""
        norm = normalize_tokens(tokenize('printf("Result: %d\\n", x)'))
        assert "STR_LIT" in norm
        assert "Result" not in norm

    # ── F.6 Preprocessor ──
    def test_include_becomes_preproc(self):
        norm = normalize_tokens(tokenize("#include <stdio.h>"))
        assert "PREPROC" in norm
        assert "stdio" not in norm

    def test_define_becomes_preproc(self):
        norm = normalize_tokens(tokenize("#define MAX 100"))
        assert "PREPROC" in norm

    # ── F.7 Operators preserved ──
    def test_operators_preserved(self):
        norm = normalize_tokens(tokenize("a == b && c != d"))
        assert "==" in norm
        assert "&&" in norm
        assert "!=" in norm

    def test_scope_resolution_preserved(self):
        norm = normalize_tokens(tokenize("std::sort"))
        assert "::" in norm


# ─────────────────────────────────────────────────────────────────────────────
# G) Tutarlılık Testleri
# ─────────────────────────────────────────────────────────────────────────────
class TestConsistency:
    def test_deterministic(self):
        """Aynı giriş her zaman aynı çıktı üretmeli"""
        code = "int fib(int n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); }"
        t1 = normalize_tokens(tokenize(code))
        t2 = normalize_tokens(tokenize(code))
        assert t1 == t2

    def test_normalize_is_subset_preserving(self):
        """normalize_tokens sonucu en az tokenize kadar uzun olmalı (1-to-1 mapping)"""
        code = "int x = 0xFF + 3.14;"
        tokens = tokenize(code)
        norm = normalize_tokens(tokens)
        assert len(norm) == len(tokens)

    def test_all_keywords_not_var_or_func(self):
        """Hiçbir keyword VAR veya FUNC'a dönüşmemeli"""
        for kw in KEYWORDS:
            norm = normalize_tokens([kw])
            assert norm[0] not in ("VAR", "FUNC", "NUM"), \
                f"Keyword '{kw}' incorrectly mapped to {norm[0]}"

    def test_empty_token_list(self):
        assert normalize_tokens([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# H) Type-2 Klon Tespiti — Kritik Testler
# ─────────────────────────────────────────────────────────────────────────────
class TestType2CloneEquivalence:
    """
    Type-2 klon tespitinin temel prensibi:
    Değişken isimleri değiştirilmiş iki kod parçası, token normalizasyonundan
    SONRA birebir aynı token dizisine sahip olmalıdır.

    Eğer 'i' → korunuyor ama 'idx' → VAR oluyorsa:
      orijinal: ['for', '(', 'int', 'i', ...)
      klon:     ['for', '(', 'int', 'VAR', ...)  ← FARKLI → yanlış!

    Doğru davranış:
      orijinal: ['for', '(', 'int', 'VAR', ...)
      klon:     ['for', '(', 'int', 'VAR', ...)  ← AYNI → doğru!
    """

    def _norm(self, code):
        return normalize_tokens(tokenize(code))

    def test_loop_var_rename_equivalent(self):
        """i → idx rename: normalize sonrası aynı dizi"""
        original = "for (int i = 0; i < n; i++) total += arr[i];"
        clone    = "for (int idx = 0; idx < num; idx++) total += arr[idx];"
        assert self._norm(original) == self._norm(clone)

    def test_variable_rename_equivalent(self):
        """result → ret_val rename: normalize sonrası aynı"""
        original = "int result = a + b; return result;"
        clone    = "int ret_val = a + b; return ret_val;"
        assert self._norm(original) == self._norm(clone)

    def test_param_rename_equivalent(self):
        """Fonksiyon parametresi rename"""
        original = "int factorial(int n) { return n <= 1 ? 1 : n * factorial(n-1); }"
        clone    = "int factorial(int num) { return num <= 1 ? 1 : num * factorial(num-1); }"
        assert self._norm(original) == self._norm(clone)

    def test_temp_rename_equivalent(self):
        """temp → tmp_var rename"""
        original = "int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;"
        clone    = "int tmp_var = arr[i]; arr[i] = arr[j]; arr[j] = tmp_var;"
        assert self._norm(original) == self._norm(clone)

    def test_hex_vs_decimal_not_equivalent(self):
        """Hex ve decimal aynı değere sahip olsa bile normalize aynı: her ikisi NUM"""
        original = "int mask = 255;"
        hex_ver  = "int mask = 0xFF;"
        # İkisi de NUM'a dönüşür → aynı normalize çıktısı
        assert self._norm(original) == self._norm(hex_ver)

    def test_string_format_rename_irrelevant(self):
        """Format string içeriği STR_LIT'e dönüşür — içerik farkı önemsiz"""
        original = 'printf("Result: %d\n", x);'
        clone    = 'printf("Output: %d\n", x);'
        # Format string farklı ama normalize aynı çıktı vermeli
        assert self._norm(original) == self._norm(clone)


# -----------------------------------------------------------------------------
# I) Yorum Soyutlama (Comment Stripping)
# -----------------------------------------------------------------------------
class TestCommentStripping:
    """Yorumlar tokenizasyondan ONCE soyutlanmali.

    code_features.py zaten yorum soyutluyor — tokenizer.py de soyutlayarak
    tutarlilik saglandi. Yoksa yorum icindeki kelimeler ('sort', 'the',
    'array') identifier token olarak sisteme sizabiliyordu.
    """

    def _norm(self, code):
        return normalize_tokens(tokenize(code))

    def test_block_comment_content_not_tokenized(self):
        """/* bubble sort */ icerisindeki kelimeler token olmamali"""
        tokens = tokenize("/* bubble sort */ int x = 5;")
        assert "bubble" not in tokens
        assert "sort" not in tokens
        assert "int" in tokens
        assert "x" in tokens

    def test_line_comment_content_not_tokenized(self):
        """// increment counter — yorum icerigi token olmamali"""
        tokens = tokenize("int x = 5; // increment counter here")
        assert "increment" not in tokens
        assert "counter" not in tokens
        assert "int" in tokens
        assert "5" in tokens

    def test_multiline_block_comment(self):
        """Cok satirli block yorum temizlenmeli"""
        code = """
        /*
         * This function computes the factorial
         * using iteration.
         */
        int factorial(int n) { return 1; }
        """
        tokens = tokenize(code)
        assert "This" not in tokens
        assert "function" not in tokens
        assert "factorial" in tokens

    def test_block_comment_with_code_keywords(self):
        """Yorum icindeki if, for gibi anahtar kelimeler sayilmamali"""
        tokens = tokenize("/* if (x > 0) { ... } */ int y = 1;")
        assert "int" in tokens
        assert tokens.count("if") == 0

    def test_comment_stripping_preserves_rest(self):
        """Yorum temizleme sonrasi kod tokenlari bozulmamali"""
        norm1 = normalize_tokens(tokenize("int x = 5;"))
        norm2 = normalize_tokens(tokenize("int x = 5; // this is a comment"))
        assert norm1 == norm2

    def test_block_comment_stripping_preserves_code(self):
        norm1 = normalize_tokens(tokenize("int x = a + b;"))
        norm2 = normalize_tokens(tokenize("int x = /* sum of */ a + b;"))
        assert norm1 == norm2

    def test_two_clones_comment_difference_normalized(self):
        """Yorum farki olan iki klon ayni normalize sonucu vermeli (Type-1)"""
        original = "int fib(int n) { return n; }"
        clone    = "int fib(int n) { /* base case */ return n; }"
        assert self._norm(original) == self._norm(clone)


# -----------------------------------------------------------------------------
# J) Sayisal Suffix (Numeric Suffix)
# -----------------------------------------------------------------------------
class TestNumericSuffix:
    """42UL, 1000LL, 3.14f gibi suffix'li literaller NUM olmali."""

    def test_unsigned_long_suffix(self):
        norm = normalize_tokens(tokenize("42UL"))
        assert norm == ["NUM"]

    def test_long_long_suffix(self):
        norm = normalize_tokens(tokenize("1000LL"))
        assert norm == ["NUM"]

    def test_float_f_suffix(self):
        norm = normalize_tokens(tokenize("3.14f"))
        assert norm == ["NUM"]

    def test_float_L_suffix(self):
        norm = normalize_tokens(tokenize("3.14L"))
        assert norm == ["NUM"]

    def test_unsigned_suffix(self):
        norm = normalize_tokens(tokenize("255U"))
        assert norm == ["NUM"]

    def test_plain_int_still_num(self):
        norm = normalize_tokens(tokenize("42"))
        assert norm == ["NUM"]

    def test_suffix_in_expression(self):
        """Suffix'li sayi ifade icinde dogru NUM'a donusmeli"""
        norm = normalize_tokens(tokenize("int x = 42UL + 3.14f;"))
        assert norm.count("NUM") == 2
