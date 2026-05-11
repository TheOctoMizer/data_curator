from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# Checkpoints produced by experiments/*.py (run from repo root).
MODEL_CHECKPOINTS: list[tuple[str, Path]] = [
    ("Non-curated", Path("outputs/noncurated-gpt2/final")),
    ("Curated", Path("outputs/curated-gpt2/final")),
    ("Curriculum", Path("outputs/curated-curriculum-gpt2/final")),
]


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prepare_eval_dataset(
    *,
    tokenizer: GPT2TokenizerFast,
    max_eval_samples: int,
    block_size: int,
):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda row: bool(row["text"] and row["text"].strip()))
    dataset = dataset.select(range(min(max_eval_samples, len(dataset))))

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing evaluation set",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def _evaluate_model_with_timing(
    *,
    model_dir: Path,
    eval_dataset,
    batch_size: int,
    device: str,
    warmup_batches: int = 2,
) -> tuple[float, float, dict[str, float]]:
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    losses: list[float] = []
    batch_times_ms: list[float] = []
    tokens_per_timed_batch: list[int] = []

    with torch.inference_mode():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            losses.append(float(outputs.loss.item()))
            if step >= warmup_batches:
                batch_times_ms.append(elapsed_ms)
                tokens_per_timed_batch.append(int(attention_mask.sum().item()))

    avg_loss = sum(losses) / max(1, len(losses))
    raw_ppl = float(math.exp(avg_loss)) if avg_loss < 80 else float("inf")
    ppl = raw_ppl if math.isfinite(raw_ppl) else float("inf")

    timing: dict[str, float] = {}
    if batch_times_ms:
        timing["eval_ms_per_batch_mean"] = sum(batch_times_ms) / len(batch_times_ms)
        timing["eval_batches_timed"] = float(len(batch_times_ms))
        total_tokens_timed = sum(tokens_per_timed_batch)
        total_sec = sum(batch_times_ms) / 1000.0
        timing["eval_tokens_per_sec"] = total_tokens_timed / max(total_sec, 1e-6)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return avg_loss, ppl, timing


def _strip_helpers(s: str, max_len: int = 220) -> str:
    s = s.replace("\n", " ").strip()
    return s[:max_len] + ("…" if len(s) > max_len else "")


def _generate_sample(
    *,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
    prompt: str,
    max_new_tokens: int,
    do_warmup: bool,
) -> tuple[str, dict[str, float]]:
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_warmup:
        with torch.inference_mode():
            _ = model.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids, attention_mask=attn, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = out.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    metrics = {
        "gen_latency_sec": elapsed,
        "gen_new_tokens": float(new_tokens),
        "gen_tokens_per_sec": new_tokens / max(elapsed, 1e-6),
    }
    return text, metrics


def main() -> None:
    """Compare checkpoints: eval loss, throughput, and greedy generations."""
    missing = [path for _, path in MODEL_CHECKPOINTS if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing checkpoint directory(ies). Train first, then re-run:\n"
            f"{lines}\n"
            "  noncurated: experiments/noncurated_training.py\n"
            "  curated: experiments/curated_training.py (after curate_dataset.py)\n"
            "  curriculum: experiments/curated_curriculum_training.py (after curate_dataset.py)"
        )

    block_size = 128
    max_eval_samples = 512
    eval_batch_size = 8
    device = _select_device()

    print(f"Using device: {device}")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    eval_dataset = _prepare_eval_dataset(
        tokenizer=tokenizer,
        max_eval_samples=max_eval_samples,
        block_size=block_size,
    )
    n_models = len(MODEL_CHECKPOINTS)
    print(
        f"Evaluating on {len(eval_dataset)} test records (same split for all {n_models} models).\n"
    )

    print("--- Teacher-forcing eval (loss on padded wikitext test batches) ---")
    label_width = max(len(name) for name, _ in MODEL_CHECKPOINTS)
    eval_rows: list[tuple[str, float, float, dict[str, float]]] = []
    for label, model_dir in MODEL_CHECKPOINTS:
        loss, ppl, timing = _evaluate_model_with_timing(
            model_dir=model_dir,
            eval_dataset=eval_dataset,
            batch_size=eval_batch_size,
            device=device,
        )
        eval_rows.append((label, loss, ppl, timing))
        print(f"{label:<{label_width}} | loss={loss:.4f} | ppl={ppl:.4f}")
        if timing:
            print(
                f"{' ' * label_width} | eval ~{timing['eval_ms_per_batch_mean']:.1f} ms/batch | "
                f"~{timing['eval_tokens_per_sec']:.0f} tokens/s (post-warmup)"
            )

    print("\n=== Interpretation ===")
    print(
        "Lower loss here means better next-token prediction on these fixed-length padded batches. "
        "If one run used many more steps or different training data, loss is not a fair single number "
        "for 'which pipeline is better'."
    )
    nc_ppl = next((ppl for lab, _, ppl, _ in eval_rows if lab == "Non-curated"), float("nan"))
    if math.isfinite(nc_ppl) and nc_ppl > 100:
        print(
            "Note: Very high non-curated perplexity often means that checkpoint did not train long enough, "
            "failed to save correctly, or does not match this tokenizer/setup."
        )

    winner: str | None = None
    best_ppl = float("inf")
    for label, _, ppl, _ in eval_rows:
        if math.isfinite(ppl) and ppl < best_ppl:
            best_ppl = ppl
            winner = label
    print("\n=== Winner on eval loss (use with caution) ===")
    if winner is not None:
        print(f"{winner} checkpoint has the lowest perplexity on this eval (~{best_ppl:.4f}).")
    else:
        print("Could not pick a winner (check for inf/overflow in loss).")

    print("\n--- Greedy generation (same prompts, max_new_tokens=80) ---")
    gen_label_width = max(len(name) for name, _ in MODEL_CHECKPOINTS)
    prompts = [
        "The history of Rome begins with",
        "In machine learning, perplexity measures",
        "Scientists discovered that",
    ]

    loaded: list[tuple[str, GPT2LMHeadModel]] = []
    for label, model_dir in MODEL_CHECKPOINTS:
        m = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device).eval()
        loaded.append((label, m))

    for i, prompt in enumerate(prompts):
        print(f"\n>>> Prompt {i + 1}: {prompt!r}")
        for j, (label, model) in enumerate(loaded):
            text, gen = _generate_sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=80,
                do_warmup=(i == 0 and j == 0),
            )
            print(f"  {label:<{gen_label_width}}: {_strip_helpers(text)}")
            print(
                f"  {' ' * gen_label_width}  [gen {int(gen['gen_new_tokens'])} tok in "
                f"{gen['gen_latency_sec'] * 1000:.1f} ms, ~{gen['gen_tokens_per_sec']:.1f} tok/s]"
            )

    for _, model in loaded:
        del model
    loaded.clear()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
