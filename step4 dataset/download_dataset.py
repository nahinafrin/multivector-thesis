"""Download the `rag-datasets/rag-mini-wikipedia` dataset and save each split as JSONL.

Outputs are written under ./data/<config>/<split>.jsonl, e.g.:

    data/question-answer/test.jsonl
    data/text-corpus/passages.jsonl

Each line in the output file is a single JSON object (one record per line).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

REPO_ID = "rag-datasets/rag-mini-wikipedia"
CONFIGS = ("question-answer", "text-corpus")


def save_split_as_jsonl(dataset: Dataset, output_path: Path) -> int:
    """Write a `datasets.Dataset` to a JSONL file. Returns the number of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(dataset)


def download_config(config_name: str, output_root: Path) -> None:
    print(f"\n=== Loading config: {config_name} ===")
    ds = load_dataset(REPO_ID, config_name)

    # `load_dataset` returns a DatasetDict (split -> Dataset). Some configs
    # only have one split (e.g. `passages` for the corpus); handle both cases.
    splits: dict[str, Dataset]
    if isinstance(ds, DatasetDict):
        splits = dict(ds)
    else:
        splits = {"train": ds}

    config_dir = output_root / config_name
    for split_name, split_ds in splits.items():
        out_file = config_dir / f"{split_name}.jsonl"
        n = save_split_as_jsonl(split_ds, out_file)
        print(f"  - {split_name}: {n} rows -> {out_file}")
        print(f"    columns: {split_ds.column_names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where JSONL files will be written (default: ./data)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIGS,
        default=list(CONFIGS),
        help="Which dataset configs to download (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        download_config(config, args.output_dir)
    print(f"\nDone. Files written under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
