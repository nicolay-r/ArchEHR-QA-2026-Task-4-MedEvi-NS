# MedEvi-NS @ ArchEHR-QA 2026 Task 4

### [📊 ArchEHR-QA 2026 Task 4 Competition](https://www.codabench.org/competitions/13528/)

### [📜 Poster](docs/poster-CL4Health.pdf)

### [📄 Paper](https://aclanthology.org/2026.cl4health-1.44/)

This repository represent a code for paper [MedEvi-NS at ArchEHR-QA 2026: Using clinical reasoning
principles to improve zero-shot capabilities of Large Language Models in Evidence Alignment](https://aclanthology.org/2026.cl4health-1.44/) @ [LREC 2026](https://www.elra.info/lrec2026/).
The task aims at citing suporting sentences from patient excerpt notes to each sentence from answers to patient question.

This is an implementation of the  prompt-engineering methodology that features **clinical-reasoning principles** (See figure below) in related alignment. 
We adopt this methodology for `GPT-5.2` in zero-shot learning mode. 

<p align="center">
  <img src="logo.png" width="300" />
</p>


# Installation

```bash
pip3 install -r dependencies.txt
```

# Usage 

## Inference

We use [`bulk-chain` project](https://github.com/nicolay-r/bulk-chain) to perform fast-inference with native support batching and async API querying.

**Supported providers:** OpenAI, Replicate.

> **More providers at:** https://github.com/nicolay-r/nlp-thirdgate

Navigate to `src` directory.

```python
python pred.py  \
    --model_name "gpt-5.2-2025-12-11"  \
    --provider_path "providers/openai_156.py"  \
    --dataset_name "train_qa_without_cite" \
    --api_token OPENAI_TOKEN_GOES_HERE
```

## Result Evaluation

Proceed with `eval.py` in scripts as follows:

> **Important:** you need to fetch data `archehr-qa_key.json` that lists reference data (citations per each answer sentence). Please follow the [Dataset](#dataset) section.

```python
python eval.py \
  --pred_path data/pred/YOUR-PRED-FILE.jsonl \
  --key_path data/orig/dev/archehr-qa_key.json
```


# Dataset

https://physionet.org/content/archehr-qa-bionlp-task-2025/1.3/


# Reference

```
@inproceedings{sun-etal-2026-medevi,
  title = {MedEvi-NS at ArchEHR-QA 2026: Using Clinical Reasoning Principles to Improve Zero-shot Capabilities of Large Language Models in Evidence Alignment },
  author = {Sun, Mengxuan and Rusnachenko, Nicolay},
  booktitle = {Proceedings of the Third Workshop on Patient-Oriented Language Processing (CL4Health) @ LREC 2026},
  month = {May},
  year = {2026},
  pages = {482--486},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Gupta, Deepak and Thompson, Paul and Ananiadou, Sophia and Demner-Fushman, Dina},
  doi = {10.63317/2uoe94adhkng},
  abstract = {The ArchEHR-QA shared task focuses on grounded question answering using patient EHR data. For the given clinical interpretation of the patient question, note excerpt (E) and answer text (A), subtask 4 (evidence alignment) aims to cite supporting sentences from E for each sentence in A. In this paper, we propose a prompt-engineering methodology that features clinical-reasoning principles in related alignment. We adopt this methodology for GPT-5.2 in zero-shot learning mode. According to our experiments on ArchEHR-QA, incorporating clinical reasoning principles into the prompt improves F 1overall by +2.0%. Our final submission resulted in 77.4% by F 1overall, which positions us at 10th out of 16 teams. Our code is publicly available: https://github.com/nicolay-r/ArchEHR-QA-2026-Task-4-MedEvi-NS}
}
```
