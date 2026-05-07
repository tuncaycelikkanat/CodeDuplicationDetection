"""
Paylaşılan benzerlik helper fonksiyonları
=========================================
Bu fonksiyonlar önceden pairing/pair_generator.py,
utils/feature_pipeline.py ve web_demo/app.py içinde
birer birer kopyalanmış haldeydi (DRY ihlali).
Tüm import'lar buradan yapılmalıdır.
"""


def _jaccard_sim(set_a, set_b):
    """Jaccard benzerliği iki küme arasında. İkisi de boşsa 1.0 döner."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 1.0


def _string_bigram_jaccard(s1, s2):
    """İki string arasında bigram Jaccard benzerliği."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    def _bg(s):
        if len(s) < 2:
            return {s}
        return set(s[i:i + 2] for i in range(len(s) - 1))

    bg1, bg2 = _bg(s1), _bg(s2)
    union = len(bg1 | bg2)
    return len(bg1 & bg2) / union if union > 0 else 1.0


def _tuple_bigram_jaccard(t1, t2):
    """İki tuple arasında bigram Jaccard benzerliği."""
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    def _bg(t):
        if len(t) < 2:
            return {t}
        return set((t[i], t[i + 1]) for i in range(len(t) - 1))

    bg1, bg2 = _bg(t1), _bg(t2)
    union = len(bg1 | bg2)
    return len(bg1 & bg2) / union if union > 0 else 1.0
