import argparse
import json
from os.path import basename
from pathlib import Path
import zipfile

from utils_parse import parse_grounded_answer
from utils_levenstein import normalized_similarity

def reference_sentences(case_id: str, answers_metadata: dict) -> list[str]:
    answer_metadata = answers_metadata[case_id]
    return [item['text'] for item in answer_metadata["clinician_answer_sentences"]]


def iter_parsed_to_orig(
    sentences_orig: list[str],
    sentences_parsed: list[dict],
) -> list[dict]:
    """
    Map each orig sentence to the best matching parsed entry (by Levenshtein similarity).
    Returns prediction list: one entry per orig, with answer_id = 1-based orig index,
    evidence_id from the matched parsed entry.
    """
    prediction = []
    for entry in sentences_parsed:
        parsed_text = (entry.get("sentence") or "").strip()
        best_j = -1
        best_sim = -1.0
        for j, orig in enumerate(sentences_orig):
            sim = normalized_similarity(orig, parsed_text)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0:
            cites = entry.get("evidence_id") or []
            citations = [int(c) for c in cites]
        else:
            citations = []
        prediction.append({"answer_id": best_j + 1, "evidence_id": citations})

    grouped = {}
    for item in prediction:
        if item["answer_id"] not in grouped:
            grouped[item["answer_id"]] = []
        grouped[item["answer_id"]].extend(item["evidence_id"])

    for answer_id in sorted(grouped.keys()):
        citations = grouped[answer_id]
        yield {"answer_id": str(answer_id), "evidence_id": [str(c) for c in sorted(set(citations))]}


def pred_jsonl_to_submission(pred_path, out_path, strict, answers_metadata) -> list:
    pred_path = Path(pred_path)
    out_path = Path(out_path)
    submission = []

    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            case_id = rec.get("case_id")
            if case_id is None:
                continue

            grounded_answer = rec.get("grounded_answer") or ""
            sentences_parsed = parse_grounded_answer(grounded_answer)

            # case_id
            sentences_orig = reference_sentences(str(case_id), answers_metadata)

            expected = len(sentences_orig)

            prediction = list(iter_parsed_to_orig(sentences_orig, sentences_parsed))

            for ind, item in enumerate(prediction):
                assert item["answer_id"] == str(ind + 1), (f"case_id={case_id}: prediction[ind]['answer_id']={prediction[ind]['answer_id']} != {ind + 1}")

            if strict and len(prediction) != expected:
                assert False, (f"case_id={case_id}: len(prediction)={len(prediction)} != {expected} (reference)")

            submission.append({
                "case_id": str(case_id),
                "prediction": prediction,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    return submission


def main():
    parser = argparse.ArgumentParser(description="Convert pred JSONL to submission.json")
    parser.add_argument("--pred_jsonl", type=Path, help="Input pred JSONL path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output submission.json path")
    parser.add_argument("--no-strict", action="store_true", help="Skip assertion on prediction vs reference sentence count")
    parser.add_argument("--answers-metadata", nargs="+")
    args = parser.parse_args()
    out = args.output or (args.pred_jsonl.parent / ".cache" / f"submission.json")

    answers_metadata = {}
    for fp in args.answers_metadata:
        with open(fp, "r", encoding="utf-8") as f:
            for item in json.load(f):
                answers_metadata[item["case_id"]] = item

    pred_jsonl_to_submission(args.pred_jsonl, out, strict=not args.no_strict, answers_metadata=answers_metadata)
    print(f"Wrote submission: {out}")

    zip_output = args.output or (args.pred_jsonl.parent / f"submission_{basename(args.pred_jsonl).replace('.jsonl', '.zip')}")
    with zipfile.ZipFile(zip_output, "w") as zipf:
        zipf.write(out, arcname=basename(out))
    print(f"Wrote submission zip: {zip_output}")


if __name__ == "__main__":
    main()
