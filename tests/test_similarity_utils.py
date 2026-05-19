"""
Unit testler: utils/similarity_utils.py
"""
import sys, os
from utils.logger import Log

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.similarity_utils import _jaccard_sim, _string_bigram_jaccard, _tuple_bigram_jaccard


class TestJaccardSim:
    def test_both_empty(self):
        assert _jaccard_sim(set(), set()) == 1.0

    def test_one_empty(self):
        assert _jaccard_sim({1, 2}, set()) == 0.0
        assert _jaccard_sim(set(), {1, 2}) == 0.0

    def test_identical(self):
        assert _jaccard_sim({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_no_overlap(self):
        assert _jaccard_sim({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        result = _jaccard_sim({1, 2, 3}, {2, 3, 4})
        # intersection=2, union=4 → 0.5
        assert abs(result - 0.5) < 1e-9


class TestStringBigramJaccard:
    def test_both_empty(self):
        assert _string_bigram_jaccard("", "") == 1.0

    def test_one_empty(self):
        assert _string_bigram_jaccard("abc", "") == 0.0
        assert _string_bigram_jaccard("", "abc") == 0.0

    def test_identical(self):
        assert _string_bigram_jaccard("hello", "hello") == 1.0

    def test_completely_different(self):
        assert _string_bigram_jaccard("ab", "cd") == 0.0

    def test_single_char(self):
        # Tek karakter → bigram kümesi = {char} olarak ele alınır
        assert _string_bigram_jaccard("a", "a") == 1.0
        assert _string_bigram_jaccard("a", "b") == 0.0

    def test_partial(self):
        result = _string_bigram_jaccard("abcd", "abce")
        # bigrams("abcd")={"ab","bc","cd"}, bigrams("abce")={"ab","bc","ce"}
        # intersection=2, union=4 → 0.5
        assert abs(result - 0.5) < 1e-9


class TestTupleBigramJaccard:
    def test_both_empty(self):
        assert _tuple_bigram_jaccard((), ()) == 1.0

    def test_one_empty(self):
        assert _tuple_bigram_jaccard(("a",), ()) == 0.0

    def test_identical(self):
        t = ("if", "for", "return")
        assert _tuple_bigram_jaccard(t, t) == 1.0

    def test_single_element(self):
        # Tek elemanlı tuple → {tuple} olarak ele alınır
        assert _tuple_bigram_jaccard(("a",), ("a",)) == 1.0

    def test_no_overlap(self):
        t1 = ("if", "for")
        t2 = ("while", "return")
        result = _tuple_bigram_jaccard(t1, t2)
        # bigrams(t1)={("if","for")}, bigrams(t2)={("while","return")} → 0
        assert result == 0.0
