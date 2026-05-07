"""
Unit testler: preprocessing/tokenizer.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.tokenizer import tokenize, normalize_tokens


class TestTokenize:
    def test_simple_expression(self):
        tokens = tokenize("int x = 5;")
        assert "int" in tokens
        assert "x" in tokens
        assert "5" in tokens
        assert "=" in tokens
        assert ";" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_multi_char_operators(self):
        tokens = tokenize("a == b && c != d")
        assert "==" in tokens
        assert "&&" in tokens
        assert "!=" in tokens

    def test_function_call(self):
        tokens = tokenize("foo(x, y)")
        assert "foo" in tokens
        assert "(" in tokens

    def test_float_number(self):
        tokens = tokenize("3.14")
        assert "3.14" in tokens


class TestNormalizeTokens:
    def test_keyword_preserved(self):
        tokens = tokenize("int x;")
        norm = normalize_tokens(tokens)
        assert "int" in norm

    def test_number_normalized(self):
        tokens = tokenize("42")
        norm = normalize_tokens(tokens)
        assert "NUM" in norm
        assert "42" not in norm

    def test_user_variable_becomes_var(self):
        tokens = tokenize("myVariable = 1;")
        norm = normalize_tokens(tokens)
        assert "VAR" in norm

    def test_function_call_becomes_func(self):
        tokens = tokenize("myFunc(x)")
        norm = normalize_tokens(tokens)
        assert "FUNC" in norm

    def test_loop_variable_preserved(self):
        # 'i', 'j', 'k' gibi döngü değişkenleri korunur
        tokens = tokenize("for (int i = 0; i < n; i++)")
        norm = normalize_tokens(tokens)
        assert "i" in norm

    def test_keyword_not_func(self):
        # 'for' kelimesi FUNC değil keyword olarak kalmalı
        tokens = tokenize("for(int i=0;")
        norm = normalize_tokens(tokens)
        assert "for" in norm
        assert "FUNC" not in norm
