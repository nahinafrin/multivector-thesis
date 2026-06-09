# rag-mini-wikipedia downloader

Small utility to download the [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) dataset (both the `question-answer` and `text-corpus` configs) and save every split as JSONL on disk.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Download both configs into `./data`:

```powershell
python download_dataset.py
```

Pick a custom output directory or only one config:

```powershell
python download_dataset.py --output-dir out --configs text-corpus
```

## Output layout

```
data/
  question-answer/
    test.jsonl        # (split names match what the Hub returns)
  text-corpus/
    passages.jsonl
```

Each line in a `.jsonl` file is one record as a JSON object — easy to stream into a RAG pipeline downstream.
