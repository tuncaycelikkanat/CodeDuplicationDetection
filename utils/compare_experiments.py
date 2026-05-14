"""
Deney Karşılaştırma Aracı — CodeDuplicationDetection
=====================================================
Birden fazla deneyin metriklerini yan yana karşılaştırır.

Kullanım:
    python utils/compare_experiments.py
    python utils/compare_experiments.py --exp-ids 54 55 56
    python utils/compare_experiments.py --metric f1_score
"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_METRICS_ORDER = ["accuracy", "f1_score", "mcc", "auc_roc", "pr_auc"]
_HIGHLIGHT = "\033[92m"  # Green
_RESET = "\033[0m"
_DIM = "\033[2m"


def _list_experiments(base_dir: str) -> List[tuple]:
    """Returns sorted list of (exp_id, exp_name, exp_path) tuples."""
    results = []
    if not os.path.exists(base_dir):
        return results
    for name in os.listdir(base_dir):
        m = re.match(r"exp_(\d+)_", name)
        if m:
            results.append((int(m.group(1)), name, os.path.join(base_dir, name)))
    return sorted(results, key=lambda x: x[0])


def _load_metrics(exp_path: str) -> Dict:
    """Load test metrics from an experiment directory."""
    metrics = {}
    test_path = os.path.join(exp_path, "metrics_test.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            metrics["test"] = json.load(f)
    train_path = os.path.join(exp_path, "metrics_train.json")
    if os.path.exists(train_path):
        with open(train_path) as f:
            metrics["train"] = json.load(f)
    config_path = os.path.join(exp_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            metrics["config"] = json.load(f)
    return metrics


def _load_timing(exp_path: str) -> Optional[float]:
    """Extract total training time from notes.txt if available."""
    notes_path = os.path.join(exp_path, "notes.txt")
    if not os.path.exists(notes_path):
        return None
    with open(notes_path) as f:
        for line in f:
            if "TOTAL" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    try:
                        return float(p)
                    except ValueError:
                        continue
    return None


def compare(exp_ids: Optional[List[int]] = None, base_dir: str = "experiments", sort_by: str = "f1_score"):
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(_PROJECT_ROOT, base_dir)

    all_exps = _list_experiments(base_dir)
    if not all_exps:
        print("❌ Hiç deney bulunamadı.")
        return

    if exp_ids:
        id_set = set(exp_ids)
        exps = [(eid, name, path) for eid, name, path in all_exps if eid in id_set]
        missing = id_set - {e[0] for e in exps}
        if missing:
            print(f"⚠️  Bulunamayan deney ID'leri: {sorted(missing)}")
    else:
        exps = all_exps

    if not exps:
        print("❌ Belirtilen deneyler bulunamadı.")
        return

    # Load data
    rows = []
    for eid, name, path in exps:
        data = _load_metrics(path)
        test_m = data.get("test", {})
        train_m = data.get("train", {})
        cfg = data.get("config", {})
        total_time = _load_timing(path)
        rows.append({
            "id": eid,
            "name": name,
            "model": cfg.get("model_name", "?"),
            "pairs": cfg.get("pair_count", "?"),
            "test": test_m,
            "train": train_m,
            "time_min": total_time,
        })

    # Sort by metric
    rows.sort(key=lambda r: r["test"].get(sort_by, 0.0), reverse=True)

    # Header
    col_w = 10
    name_w = 40
    print(f"\n{'='*100}")
    print(f"  DENEY KARŞILAŞTIRMA  ({len(rows)} deney)  |  Sıralama: {sort_by}")
    print(f"{'='*100}")

    header = f"{'ID':<5} {'Deney Adı':<{name_w}} {'Model':<15} {'Pairs':>8}"
    for m in _METRICS_ORDER:
        header += f"  {m[:8]:>{col_w}}"
    header += f"  {'Süre(dk)':>9}"
    print(header)
    print("-" * len(header))

    best_vals = {m: max(r["test"].get(m, 0.0) for r in rows) for m in _METRICS_ORDER}

    for r in rows:
        line = f"{r['id']:<5} {r['name'][:name_w]:<{name_w}} {r['model'][:15]:<15} {str(r['pairs']):>8}"
        for m in _METRICS_ORDER:
            val = r["test"].get(m)
            if val is None:
                cell = f"{'N/A':>{col_w}}"
            else:
                formatted = f"{val:.4f}"
                if val == best_vals.get(m):
                    cell = f"{_HIGHLIGHT}{formatted:>{col_w}}{_RESET}"
                else:
                    cell = f"{formatted:>{col_w}}"
            line += f"  {cell}"
        time_str = f"{r['time_min']:.1f}" if r["time_min"] else "N/A"
        line += f"  {time_str:>9}"
        print(line)

    print(f"{'='*100}")
    best = rows[0]
    print(f"\n🏆 En iyi deney (by {sort_by}): {_HIGHLIGHT}{best['name']}{_RESET}")
    print(f"   Test F1={best['test'].get('f1_score', 0):.4f}  "
          f"MCC={best['test'].get('mcc', 0):.4f}  "
          f"AUC-ROC={best['test'].get('auc_roc', 0):.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deney metriklerini karşılaştır.")
    parser.add_argument("--exp-ids", type=int, nargs="+", default=None,
                        help="Karşılaştırılacak deney ID'leri (örn. 54 55 56). "
                             "Belirtilmezse tüm deneyler listelenir.")
    parser.add_argument("--metric", type=str, default="f1_score",
                        choices=_METRICS_ORDER,
                        help="Sıralama metriği (varsayılan: f1_score)")
    parser.add_argument("--base-dir", type=str, default="experiments",
                        help="Deney klasörü (varsayılan: experiments/)")
    args = parser.parse_args()
    compare(exp_ids=args.exp_ids, base_dir=args.base_dir, sort_by=args.metric)
