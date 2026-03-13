import json
from os import makedirs
from os.path import abspath, dirname, join

def write_jsonl(results_it, output_path):
    makedirs(dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results_it:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def iter_cases(file_path):
    """Yield case dicts from a JSONL file (one per line) or a single JSON file (one object or list)."""
    cur_dir = dirname(abspath(__file__))
    path = join(cur_dir, "..", file_path) if not file_path.startswith("/") else file_path
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if not raw.strip():
        return
    first = raw.lstrip()[0]
    if first == "[":
        for case in json.loads(raw):
            yield case
    elif first == "{":
        yield json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
