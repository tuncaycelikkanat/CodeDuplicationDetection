from utils.logger import Log
"""
Paylaşılan benzerlik helper fonksiyonları
=========================================
Bu fonksiyonlar önceden pairing/pair_generator.py,
utils/feature_pipeline.py ve web_demo/app.py içinde
birer birer kopyalanmış haldeydi (DRY ihlali).
Tüm import'lar buradan yapılmalıdır.
"""


def _jaccard_sim(obj_a, obj_b):
    """
    Jaccard benzerliği iki küme veya multiset (dict/Counter) arasında.
    İkisi de boşsa 1.0 döner.
    """
    if not obj_a and not obj_b:
        return 1.0
    if not obj_a or not obj_b:
        return 0.0
        
    if isinstance(obj_a, (set, frozenset)):
        union = len(obj_a | obj_b)
        return len(obj_a & obj_b) / union if union > 0 else 1.0
    elif isinstance(obj_a, dict):
        # Multiset Jaccard: sum(min(A, B)) / sum(max(A, B))
        keys = set(obj_a.keys()) | set(obj_b.keys())
        intersection_sum = sum(min(obj_a.get(k, 0), obj_b.get(k, 0)) for k in keys)
        union_sum = sum(max(obj_a.get(k, 0), obj_b.get(k, 0)) for k in keys)
        return intersection_sum / union_sum if union_sum > 0 else 1.0
    
    return 0.0


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
