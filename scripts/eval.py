#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils_data import parse_grounded_answer


def load_key_links(key_path: Path, positive_labels: Set[str]) -> Dict[int, Set[int]]:

    with open(key_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold: Dict[int, Set[int]] = {}
    for item in data:
        cid_raw = item.get("case_id")
        if cid_raw is None:
            continue
        try:
            cid = int(cid_raw)
        except (TypeError, ValueError):
            continue

        links: Set[int] = set()
        for ans in item.get("answers", []):
            sid_raw = ans.get("sentence_id")
            rel = ans.get("relevance")
            if rel not in positive_labels or sid_raw is None:
                continue
            try:
                sid = int(sid_raw)
            except (TypeError, ValueError):
                continue
            links.add(sid)

        gold[cid] = links

    return gold


def load_pred_links(pred_path: Path) -> Dict[int, Set[int]]:
    """
    Load predicted alignments from the JSONL produced by solution.py.

    Expected structure per line:
    {
      "case_id": 21,
      ...,
      "grounded_entries": [
        {"sentence": "...", "evidence_id": ["4"]},
        {"sentence": "...", "evidence_id": ["5", "11"]},
        ...
      ]
    }
    """
    preds: Dict[int, Set[int]] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid_raw = record.get("case_id")
            if cid_raw is None:
                continue
            try:
                cid = int(cid_raw)
            except (TypeError, ValueError):
                continue

            links: Set[int] = set()

            # TODO. apply 
            grounded_entries = parse_grounded_answer(record.get("grounded_answer"))

            for entry in grounded_entries:
                ids = entry.get("evidence_id") or []
                for sid_raw in ids:
                    try:
                        # evidence ids are usually strings like "4"
                        sid = int(str(sid_raw).strip())
                    except (TypeError, ValueError):
                        continue
                    links.add(sid)

            preds[cid] = links

    return preds


def safe_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, F1 from counts."""
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate(
    gold: Mapping[int, Set[int]],
    preds: Mapping[int, Set[int]],
) -> None:
    case_ids = sorted(set(gold.keys()) & set(preds.keys()))
    if not case_ids:
        print("No overlapping case_ids between gold and predictions.")
        return

    micro_tp = micro_fp = micro_fn = 0
    macro_p = macro_r = macro_f1 = 0.0

    print(f"Evaluating {len(case_ids)} cases...")

    for cid in case_ids:
        g = gold.get(cid, set())
        p = preds.get(cid, set())

        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        scores = safe_prf(tp, fp, fn)

        print (f"Case {cid}: {scores}")
        macro_p += scores["precision"]
        macro_r += scores["recall"]
        macro_f1 += scores["f1"]

    micro = safe_prf(micro_tp, micro_fp, micro_fn)
    n = len(case_ids)
    macro = {
        "precision": macro_p / n,
        "recall": macro_r / n,
        "f1": macro_f1 / n,
    }

    print("\nMicro-averaged scores (over all links):")
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall:    {micro['recall']:.4f}")
    print(f"  F1:        {micro['f1']:.4f}")

    print("\nMacro-averaged scores (mean over cases):")
    print(f"  Precision: {macro['precision']:.4f}")
    print(f"  Recall:    {macro['recall']:.4f}")
    print(f"  F1:        {macro['f1']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounding predictions against gold alignments."
    )
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument(
        "--key_path",
        type=Path,
        required=True,
        help="Path to archehr-qa_key.json (dev/test key).",
    )
    parser.add_argument(
        "--positive_labels",
        type=str,
        default="essential,supplementary",
        help=(
            "Comma-separated relevance labels to treat as positive "
            '(default: "essential,supplementary").'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    pred_path = args.pred_path
    if not pred_path.is_absolute():
        pred_path = project_root / pred_path

    key_path = args.key_path
    if not key_path.is_absolute():
        key_path = project_root / key_path

    positive_labels = {s.strip() for s in args.positive_labels.split(",") if s.strip()}

    gold = load_key_links(key_path, positive_labels)
    preds = load_pred_links(pred_path)

    evaluate(gold, preds)


if __name__ == "__main__":
    main()

