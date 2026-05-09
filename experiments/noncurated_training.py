from datasets import load_dataset
from pathlib import Path
import shutil
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    """Train GPT-2 on non-curated data using only datasets library."""
    model_name = "gpt2"
    block_size = 128
    batch_size = 4
    learning_rate = 5e-5
    num_train_epochs = 1
    seed = 42
    output_dir = Path("outputs/noncurated-gpt2")

    print("Loading raw train split (datasets only)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda row: bool(row["text"] and row["text"].strip()))
    print(f"Raw records: {len(dataset)}")

    print("Creating train/validation/test split...")
    train_val_test = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=seed)
    splits = {
        "train": train_val_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }
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
        remove_columns=["text"],
        desc="Tokenizing non-curated train",
    )
    tokenized_validation = splits["validation"].map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing non-curated validation",
    )
    tokenized_test = splits["test"].map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing non-curated test",
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

    print("Starting non-curated GPT-2 training...")
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
    print("Non-curated training complete.")


if __name__ == "__main__":
    main()