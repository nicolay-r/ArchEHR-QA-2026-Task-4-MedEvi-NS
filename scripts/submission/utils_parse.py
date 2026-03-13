import re

def _expand_citation_bracket(bracket_content: str) -> list[str]:
    """
    Parse bracket content like "1, 2, 4" or "19-25" or "1, 3-5, 7" into a list of evidence ids (strings).
    Supports ranges: "a-b" expands to a, a+1, ..., b inclusive.
    """
    citations = []
    for part in bracket_content.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
            if range_match:
                lo, hi = int(range_match.group(1)), int(range_match.group(2))
                if lo <= hi:
                    citations.extend(str(i) for i in range(lo, hi + 1))
        elif part.isdigit():
            citations.append(part)

    return citations


def parse_grounded_answer(text: str, drop_content_after_last_newline: bool = False) -> list[dict]:
    if not text or not text.strip():
        return []
    raw = text.strip()
    for prefix in ("Here is the output:", "Output:", "Output "):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    if drop_content_after_last_newline and "\n\n" in raw:
        raw = raw[: raw.rfind("\n\n")].rstrip()

    entries = []

    # If no citation brackets found, fall back to line-by-line (legacy)
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"\s*\[([^\]]*)\]\s*\.?\s*$", line)
        if not m:
            entries.append({"sentence": line, "evidence_id": []})
            continue
        bracket_content = m.group(1).strip()
        sentence = line[: m.start()].strip().rstrip(".")
        citations = _expand_citation_bracket(bracket_content)
        entries.append({"sentence": sentence, "evidence_id": citations})

    return entries
