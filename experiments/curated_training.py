"""Train GPT-2 on a curated dataset produced by curate_dataset.py.

Run curation first:

    python experiments/curate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import shutil

from datasets import load_from_disk
from datacurator import DataCurator
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

# Must match CURATED_OUTPUT_DIR in curate_dataset.py
CURATED_DATASET_DIR = Path("outputs/curated_wikitext_train")


def main() -> None:
    """Train GPT-2 on curated data loaded from disk."""
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
    output_dir = Path("outputs/curated-gpt2")

    print(f"Loading curated dataset from {CURATED_DATASET_DIR.resolve()}...")
    curated_dataset = load_from_disk(str(CURATED_DATASET_DIR))
    print(f"Curated dataset size: {len(curated_dataset)}")

    datacurator = DataCurator()
    print(datacurator.describe())

    print("Creating train/validation/test split...")
    splits = datacurator.split_dataset(
        curated_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_by_difficulty=True,
        seed=seed,
    )
    print(
        f"Split sizes -> train: {len(splits['train'])}, "
        f"validation: {len(splits['validation'])}, test: {len(splits['test'])}"
    )

    print("Loading GPT-2 tokenizer/model...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    tokenized_train = splits["train"].map(
        tokenize,
        batched=True,
        remove_columns=["text", "difficulty_label", "difficulty_score", "perplexity"],
        desc="Tokenizing curated train",
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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        seed=seed,
    )

    print("Starting curated GPT-2 training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=data_collator,
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
    print("Curated training complete.")


if __name__ == "__main__":
    main()
