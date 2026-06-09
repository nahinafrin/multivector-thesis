"""Download the xTRam1/safe-guard-prompt-injection dataset from Hugging Face.

Loads the dataset, prints a quick summary, and saves each split to disk as
both a Hugging Face `save_to_disk` snapshot and a CSV file for convenience.
"""

from pathlib import Path

from datasets import load_dataset


DATASET_ID = "xTRam1/safe-guard-prompt-injection"
OUTPUT_DIR = Path(__file__).parent / "safe-guard-prompt-injection"


def main() -> None:
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)

    print("\nDataset summary:")
    print(ds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hf_snapshot_dir = OUTPUT_DIR / "hf"
    print(f"\nSaving Hugging Face snapshot to: {hf_snapshot_dir}")
    ds.save_to_disk(str(hf_snapshot_dir))

    for split_name, split in ds.items():
        csv_path = OUTPUT_DIR / f"{split_name}.csv"
        print(f"Writing {split_name} split ({len(split):,} rows) -> {csv_path}")
        split.to_csv(str(csv_path), index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
