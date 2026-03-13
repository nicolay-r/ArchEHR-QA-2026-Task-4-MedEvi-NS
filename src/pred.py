import argparse
import time
from os.path import abspath, dirname, join
from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content
from tqdm import tqdm

from utils_data import write_jsonl, iter_cases
from utils_prompt import GROUNDING_PROMPT


def build_numbered_excerpt(note_excerpt_sentences):
    """Build '[id] text' lines from note_excerpt_sentences.sentences."""
    if not note_excerpt_sentences or "sentences" not in note_excerpt_sentences:
        return ""
    lines = []
    for s in note_excerpt_sentences["sentences"]:
        sid = s.get("id", "")
        text = (s.get("text") or "").strip()
        lines.append(f"{sid}. {text}")
    return " ".join(lines) + "\n"


def case_to_sample(case_dict):
    numbered = build_numbered_excerpt(case_dict.get("note_excerpt_sentences"))
    reference = case_dict.get("reference_answer") or case_dict.get("clinician_answer") or ""
    return {
        "case_id": case_dict.get("case_id"),
        "clinician_question": (case_dict.get("clinician_question") or "").strip(),
        "clinical_note_excerpt_numbered": numbered,
        "reference_answer": reference.strip(),
    }


def predict(samples, model_name, provider_path, max_retries=3, batch_size=10, sleep_time=None, **kwargs):
    print(f"[INFO] Using model: {model_name} from `{provider_path}`")

    content_it = iter_content(
        schema=[
            {
                "prompt": GROUNDING_PROMPT,
                "out": "grounded_answer",
            },
        ],
        llm=dynamic_init(class_filepath=provider_path)(
            model_name=model_name,
            assistant="You are a helpful assistant.",
            **kwargs,
        ),
        infer_mode="batch_async",
        batch_size=batch_size,
        input_dicts_it=tqdm(samples, desc=f"Predicting `{model_name}`"),
        attempts=max_retries,
    )

    for batch in content_it:
        assert isinstance(batch, list)
        for record in batch:
            yield record

        if sleep_time is not None:
            time.sleep(sleep_time)


if __name__ == "__main__":
    cur_dir = dirname(abspath(__file__))
    datasets = {
        "case_1": "data/case_1.json",
        "train_qa_without_cite": "data/train_qa_without_cite.json",
        "test_qa": "data/test_qa.json",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=None, help="Base URL")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--provider_path", type=str, default="providers/replicate_104.py", help="Provider path")
    parser.add_argument("--api_token", type=str, default=None, help="API key")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--sleep_time", type=int, default=None, help="Sleep time")
    parser.add_argument("--dataset_name", type=str, default="train_qa_10", help="Dataset name", choices=list(datasets.keys())
    )
    args = parser.parse_args()

    args_dict = dict(vars(args))
    del args_dict["dataset_name"]

    dataset_path = datasets[args.dataset_name]
    if not dataset_path.startswith("/"):
        dataset_path = join(cur_dir, "..", dataset_path)

    def sample_it():
        for case in iter_cases(dataset_path):
            yield case_to_sample(case)


    result_template = f"pred_{args.dataset_name}_{args.model_name.split('/')[-1]}.jsonl"
    output_path = join(cur_dir, "..", "data", "pred", result_template)

    results_it = predict(samples=sample_it(), **args_dict)
    write_jsonl(results_it=results_it, output_path=output_path)
