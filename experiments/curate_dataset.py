"""Curate a dataset with DataCurator and save it to disk for curated_training.py.

Run once (or when you want to refresh labels). Then train with:

    python experiments/curated_training.py

Paths are relative to the current working directory (run from repo root recommended).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from datacurator import DataCurator

# Must match CURATED_DATASET_DIR in curated_training.py
CURATED_OUTPUT_DIR = Path("outputs/curated_wikitext_train")
SPILL_DIR = Path("outputs/curated_spill")
SPILL_CHUNK_SIZE = 500


def main() -> None:
    curator = DataCurator()
    print(curator.describe())

    loaded = curator.load_model()
    print(loaded.model_id, loaded.device, loaded.quantization)

    source = curator.stream_dataset(
        "wikitext",
        config="wikitext-2-raw-v1",
        split="train",
    )

    print("Curating (perplexity + difficulty); writing spill shards as needed...")
    curated = curator.create_difficulty_dataset(
        source,
        loaded_model=loaded,
        text_key="text",
        limit=None,
        show_progress=True,
        spill_to_disk=True,
        spill_dir=str(SPILL_DIR),
        spill_chunk_size=SPILL_CHUNK_SIZE,
        unload_source_dataset=True,
        unload_model_after=True,
    )
    print(f"Curated rows: {len(curated)}")

    if CURATED_OUTPUT_DIR.exists():
        shutil.rmtree(CURATED_OUTPUT_DIR)
    CURATED_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    curated.save_to_disk(str(CURATED_OUTPUT_DIR))
    print(f"Saved curated dataset to {CURATED_OUTPUT_DIR.resolve()}")
    print("Next: python experiments/curated_training.py")


if __name__ == "__main__":
    main()
