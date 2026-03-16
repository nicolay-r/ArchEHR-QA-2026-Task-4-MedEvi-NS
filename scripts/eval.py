#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from submission.utils_parse import parse_grounded_answer


def _citations_str_to_set(citations_str: str) -> Set[int]:
    """Parse citations string (e.g. '2', '2,5', '19-25') into set of ints."""
    if not citations_str or not str(citations_str).strip():
        return set()
    out: Set[int] = set()
    for part in str(citations_str).strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if lo <= hi:
                    out.update(range(lo, hi + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def load_key_sentence_links(key_path: Path) -> Dict[int, List[Set[int]]]:
    """
    Load gold citation sets per answer sentence from archehr-qa_key.json.
    Returns case_id -> list of sets, one set per clinician_answer_sentences item (citations field).
    """
    with open(key_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold: Dict[int, List[Set[int]]] = {}
    for item in data:
        cid_raw = item.get("case_id")
        if cid_raw is None:
            continue
        try:
            cid = int(cid_raw)
        except (TypeError, ValueError):
            continue

        sentence_links: List[Set[int]] = []
        for sent in item.get("clinician_answer_sentences") or []:
            if not isinstance(sent, dict):
                continue
            raw = sent.get("citations")
            sentence_links.append(_citations_str_to_set(str(raw) if raw is not None else ""))
        if sentence_links:
            gold[cid] = sentence_links

    return gold


def load_pred_sentence_links(pred_path: Path) -> Dict[int, List[Set[int]]]:
    preds: Dict[int, List[Set[int]]] = {}
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

            sentence_links: List[Set[int]] = []
            for entry in parse_grounded_answer(record.get("grounded_answer")):
                ids = entry.get("evidence_id") or []
                s: Set[int] = set()
                for sid_raw in ids:
                    try:
                        s.add(int(str(sid_raw).strip()))
                    except (TypeError, ValueError):
                        continue
                sentence_links.append(s)
            preds[cid] = sentence_links

    return preds


def safe_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, F1 from counts."""
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate(
    gold: Mapping[int, List[Set[int]]],
    preds: Mapping[int, List[Set[int]]],
) -> Dict[str, Any]:
    case_ids = sorted(set(gold.keys()) & set(preds.keys()))
    if not case_ids:
        return {
            "n_cases": 0,
            "case_scores": [],
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "message": "No overlapping case_ids between gold and predictions.",
        }

    case_scores: List[Dict[str, Any]] = []
    micro_tp = micro_fp = micro_fn = 0
    all_sentence_f1: List[float] = []

    for cid in case_ids:
        gold_sents = gold.get(cid, [])
        pred_sents = preds.get(cid, [])
        n_sent = min(len(gold_sents), len(pred_sents))
        if n_sent == 0:
            case_scores.append({"case_id": cid, "precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue

        case_tp = case_fp = case_fn = 0
        case_f1s: List[float] = []

        for j in range(n_sent):
            g = gold_sents[j]
            p = pred_sents[j]
            tp = len(g & p)
            fp = len(p - g)
            fn = len(g - p)
            case_tp += tp
            case_fp += fp
            case_fn += fn
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            scores = safe_prf(tp, fp, fn)
            case_f1s.append(scores["f1"])
            all_sentence_f1.append(scores["f1"])

        case_prf = safe_prf(case_tp, case_fp, case_fn)
        case_scores.append({"case_id": cid, **case_prf})

    micro = safe_prf(micro_tp, micro_fp, micro_fn)
    n_cases = len(case_ids)
    macro_f1 = sum(all_sentence_f1) / len(all_sentence_f1) if all_sentence_f1 else 0.0
    macro_p = sum(s["precision"] for s in case_scores) / n_cases if case_scores else 0.0
    macro_r = sum(s["recall"] for s in case_scores) / n_cases if case_scores else 0.0
    macro = {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
    }

    return {
        "n_cases": n_cases,
        "case_scores": case_scores,
        "micro": micro,
        "macro": macro,
        "message": None,
    }


def print_evaluate_result(result: Dict[str, Any]) -> None:
    """Print the result returned by evaluate()."""
    if result.get("message"):
        print(result["message"])
        return

    n = result["n_cases"]
    print(f"Evaluating {n} cases...")

    for item in result["case_scores"]:
        cid = item["case_id"]
        scores = {k: item[k] for k in ("precision", "recall", "f1")}
        print(f"Case {cid}: {scores}")

    micro = result["micro"]
    print("\nMicro-averaged scores (over all links, all sentences):")
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall:    {micro['recall']:.4f}")
    print(f"  F1:        {micro['f1']:.4f}")

    macro = result["macro"]
    print("\nMacro-averaged scores (mean over answer sentences):")
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

    gold = load_key_sentence_links(key_path)
    preds = load_pred_sentence_links(pred_path)
    result = evaluate(gold, preds)

    print_evaluate_result(result)


if __name__ == "__main__":
    main()

