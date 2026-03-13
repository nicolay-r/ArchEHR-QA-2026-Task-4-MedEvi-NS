import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "../scripts/submission"))
from utils_parse import parse_grounded_answer

answer = "Here is the output:\n\nThe patient has advanced hepatic encephalopathy [3, 7].\nHe has decompensated alcoholic cirrhosis [1, 13].\nHis liver disease is not suitable for transplant [15].\nHe has had multiple hospital admissions [1].\nHis doctor has discussed his prognosis with him, indicating it could be days, weeks, or months, but not likely a year [no direct support in the clinical note excerpt, but the context of the note and the discussion of goals of care and comfort measures imply a poor prognosis].\nThe patient's current condition is refractory to traditional treatment [4].\nHe has developed anuric renal failure and hepato-renal syndrome [5].\nHis goals of care have been transitioned to comfort measures [6, 12]."

entries = parse_grounded_answer(answer)
print(json.dumps(entries, indent=2))
