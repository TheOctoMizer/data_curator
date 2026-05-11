"""Train GPT-2 with difficulty-based curriculum sampling (easy → hard over progress).

Requires a curated dataset on disk (same as ``curated_training.py``):

    python experiments/curate_dataset.py
    python experiments/curated_curriculum_training.py

Unlike ``curated_training.py``, this uses :class:`datacurator.CurriculumIterableDataset`
via ``DataCurator.build_curriculum_iterable_dataset`` so each step samples by difficulty
weights from ``CurriculumSchedule.default()`` (interpolated over training progress).

Validation and test use the same tokenized map-style splits as the other experiment scripts.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

from datasets import load_from_disk
from datacurator import CurriculumSchedule, DataCurator, curriculum_lm_collate_fn
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

CURATED_DATASET_DIR = Path("outputs/curated_wikitext_train")


def main() -> None:
    if not CURATED_DATASET_DIR.exists():
        print(
            f"Missing curated dataset at {CURATED_DATASET_DIR.resolve()}. "
            "Run first:\n  python experiments/curate_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name = "gpt2"
    block_size = 128
    batch_size = 4
    learning_rate = 5e-5
    num_train_epochs = 1
    seed = 42
    output_dir = Path("outputs/curated-curriculum-gpt2")

    print(f"Loading curated dataset from {CURATED_DATASET_DIR.resolve()}...")
    curated_dataset = load_from_disk(str(CURATED_DATASET_DIR))

    datacurator = DataCurator()
    print(datacurator.describe())

    print("Creating train/validation/test split (stratified by difficulty)...")
    splits = datacurator.split_dataset(
        curated_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_by_difficulty=True,
        seed=seed,
    )
    train_split = splits["train"]
    print(
        f"Split sizes -> train: {len(train_split)}, "
        f"validation: {len(splits['validation'])}, test: {len(splits['test'])}"
    )

    print("Loading GPT-2 tokenizer/model...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    # One full schedule pass over ~len(train) curriculum draws per "epoch" of training.
    steps_per_epoch = len(train_split)
    train_iterable = datacurator.build_curriculum_iterable_dataset(
        train_split,
        tokenizer=tokenizer,
        text_key="text",
        total_epochs=num_train_epochs,
        steps_per_epoch=steps_per_epoch,
        max_length=block_size,
        seed=seed,
        schedule=CurriculumSchedule.default(),
    )
    total_samples = num_train_epochs * steps_per_epoch
    max_steps = max(1, math.ceil(total_samples / batch_size))
    print(
        f"Curriculum: total_samples={total_samples} (schedule progress 0→1), "
        f"max_steps={max_steps} at batch_size={batch_size}"
    )

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    tokenized_validation = splits["validation"].map(
        tokenize,
        batched=True,
        remove_columns=["text", "difficulty_label", "difficulty_score", "perplexity"],
        desc="Tokenizing curated validation",
    )
    tokenized_test = splits["test"].map(
        tokenize,
        batched=True,
        remove_columns=["text", "difficulty_label", "difficulty_score", "perplexity"],
        desc="Tokenizing curated test",
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    eval_steps = min(max(50, max_steps // 5), max_steps)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        remove_unused_columns=False,
    )

    print("Starting curriculum GPT-2 training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_iterable,
        eval_dataset=tokenized_validation,
        data_collator=curriculum_lm_collate_fn,
    )
    train_result = trainer.train()
    print(f"Train loss: {train_result.training_loss:.4f}")
    validation_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print(f"Validation loss: {validation_metrics.get('eval_loss', float('nan')):.4f}")
    print(f"Test loss: {test_metrics.get('test_loss', float('nan')):.4f}")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Curriculum training complete.")


if __name__ == "__main__":
    main()
