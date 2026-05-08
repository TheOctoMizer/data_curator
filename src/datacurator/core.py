"""Core module API for DataCurator."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import gc
import logging
import math
import random
from typing import Any

import torch


@dataclass(slots=True)
class LoadedModel:
    """Container for a loaded text-generation model and tokenizer."""

    model: Any
    tokenizer: Any
    model_id: str
    quantization: str
    device: str


@dataclass(slots=True)
class CurriculumSchedule:
    """Piecewise-linear curriculum schedule over training progress."""

    # Each point is (progress_in_[0,1], (easy, medium, hard))
    points: list[tuple[float, tuple[float, float, float]]]

    @classmethod
    def default(cls) -> "CurriculumSchedule":
        return cls(
            points=[
                (0.0, (0.80, 0.15, 0.05)),
                (0.5, (0.40, 0.40, 0.20)),
                (1.0, (0.05, 0.15, 0.80)),
            ]
        )

    def weights_at_progress(self, progress: float) -> tuple[float, float, float]:
        """Return interpolated (easy, medium, hard) weights for progress in [0, 1]."""
        if not self.points:
            raise ValueError("CurriculumSchedule requires at least one point.")

        p = max(0.0, min(1.0, progress))
        points = sorted(self.points, key=lambda item: item[0])
        if p <= points[0][0]:
            return _normalize_weights(points[0][1])
        if p >= points[-1][0]:
            return _normalize_weights(points[-1][1])

        for idx in range(len(points) - 1):
            start_p, start_w = points[idx]
            end_p, end_w = points[idx + 1]
            if start_p <= p <= end_p:
                alpha = 0.0 if end_p == start_p else (p - start_p) / (end_p - start_p)
                mixed = (
                    (1.0 - alpha) * start_w[0] + alpha * end_w[0],
                    (1.0 - alpha) * start_w[1] + alpha * end_w[1],
                    (1.0 - alpha) * start_w[2] + alpha * end_w[2],
                )
                return _normalize_weights(mixed)
        return _normalize_weights(points[-1][1])


class CurriculumIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that samples easy/medium/hard with dynamic curriculum weights."""

    def __init__(
        self,
        *,
        dataset: Any,
        tokenizer: Any,
        class_indices: dict[str, list[int]],
        text_key: str,
        total_steps: int,
        max_length: int,
        seed: int,
        schedule: CurriculumSchedule,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.class_indices = class_indices
        self.text_key = text_key
        self.total_steps = max(1, total_steps)
        self.max_length = max_length
        self.seed = seed
        self.schedule = schedule

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed)
        labels = ["easy", "medium", "hard"]
        available_labels = [label for label in labels if self.class_indices.get(label)]
        if not available_labels:
            raise ValueError("No samples available in curriculum class buckets.")

        for step in range(self.total_steps):
            progress = 0.0 if self.total_steps == 1 else step / (self.total_steps - 1)
            easy_w, medium_w, hard_w = self.schedule.weights_at_progress(progress)
            weights_map = {"easy": easy_w, "medium": medium_w, "hard": hard_w}
            label_pool = [label for label in labels if self.class_indices.get(label)]
            label_weights = [weights_map[label] for label in label_pool]
            chosen_label = rng.choices(label_pool, weights=label_weights, k=1)[0]
            row_index = rng.choice(self.class_indices[chosen_label])
            row = self.dataset[row_index]

            text = row.get(self.text_key, "")
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            yield {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "difficulty_label": chosen_label,
                "perplexity": float(row.get("perplexity", 0.0)),
            }

    def __len__(self) -> int:
        """Return planned number of curriculum sampling steps."""
        return self.total_steps


class DataCurator:
    """Library object for loading and streaming Hugging Face datasets."""

    def __init__(
        self,
        *,
        log_level: int = logging.INFO,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize DataCurator with structured logging."""
        self.logger = logger or logging.getLogger("datacurator")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                )
            )
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def describe(self) -> str:
        """Describe this library instance."""
        return "DataCurator helps load and stream datasets from Hugging Face."

    def load_dataset(
        self,
        repo_id: str,
        *,
        config: str | None = None,
        split: str | None = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset from Hugging Face Hub.

        Args:
            repo_id: Hugging Face dataset id, e.g. ``"imdb"``.
            config: Optional dataset config/subset name.
            split: Optional split like ``"train"`` or ``"test"``.
            streaming: If true, stream data instead of downloading locally.
            **kwargs: Extra keyword args passed to ``datasets.load_dataset``.
        """
        datasets = _import_datasets_module()
        self.logger.info(
            "Loading dataset repo_id=%s config=%s split=%s streaming=%s",
            repo_id,
            config,
            split,
            streaming,
        )
        try:
            dataset = datasets.load_dataset(
                path=repo_id,
                name=config,
                split=split,
                streaming=streaming,
                **kwargs,
            )
            self.logger.info("Dataset loaded successfully: %s", repo_id)
            return dataset
        except Exception as exc:
            self.logger.exception("Dataset load failed for repo_id=%s", repo_id)
            raise RuntimeError(
                "Failed to load dataset from Hugging Face Hub. "
                "Check network/proxy access and authentication (HF_TOKEN) if needed."
            ) from exc

    def stream_dataset(
        self,
        repo_id: str,
        *,
        config: str | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> Any:
        """Stream a dataset split from Hugging Face without full download."""
        self.logger.info(
            "Streaming dataset repo_id=%s config=%s split=%s",
            repo_id,
            config,
            split,
        )
        return self.load_dataset(
            repo_id=repo_id,
            config=config,
            split=split,
            streaming=True,
            **kwargs,
        )

    def iter_rows(
        self,
        dataset: Any,
        *,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate rows from a loaded or streaming Hugging Face dataset."""
        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            yield row

    def load_model(
        self,
        *,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a causal LM with an efficiency-first default model.

        Strategy:
        - CUDA + bitsandbytes: 4-bit NF4 quantization (best efficiency).
        - Otherwise: float16/bfloat16 fallback with automatic device mapping.
        """
        torch = _import_torch_module()
        transformers = _import_transformers_module()
        self.logger.info("Loading model model_id=%s", model_id)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        model_kwargs.update(kwargs)

        quantization = "none"
        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            model_kwargs.setdefault("dtype", torch.float16)
            if _has_bitsandbytes():
                model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                quantization = "4bit-nf4"
            else:
                quantization = "fp16"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            model_kwargs.setdefault("dtype", torch.bfloat16)
            # quantization = "fp16"
            quantization = "bf16"
        else:
            device = "cpu"
            model_kwargs.setdefault("dtype", torch.float32)
            quantization = "fp32"

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs,
            )
        except Exception as exc:
            self.logger.exception("Model load failed for model_id=%s", model_id)
            raise RuntimeError(
                "Failed to load model from Hugging Face Hub. Check model id, "
                "network/proxy configuration, and authentication (HF_TOKEN) if needed."
            ) from exc

        self.logger.info(
            "Model loaded model_id=%s device=%s quantization=%s",
            model_id,
            device,
            quantization,
        )
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            quantization=quantization,
            device=device,
        )

    def stream_perplexities(
        self,
        dataset: Any,
        *,
        loaded_model: LoadedModel,
        text_key: str = "text",
        limit: int | None = None,
        max_length: int = 512,
        show_progress: bool = True,
        progress_desc: str = "Computing perplexity",
        log_every: int = 100,
        medium_threshold: float = 20.0,
        hard_threshold: float = 60.0,
    ) -> Iterator[dict[str, Any]]:
        """Stream rows and compute perplexity per record.

        Yields dictionaries containing:
        - index: row index in the processed stream
        - text: the evaluated text
        - perplexity: float perplexity value
        - difficulty_score: normalized score in [0.0, 1.0]
        - difficulty_label: easy/medium/hard
        - record: original row dict
        """
        torch = _import_torch_module()
        tqdm = _import_tqdm()

        model = loaded_model.model
        tokenizer = loaded_model.tokenizer

        self.logger.info(
            "Starting perplexity stream model_id=%s text_key=%s limit=%s max_length=%s medium_threshold=%.3f hard_threshold=%.3f",
            loaded_model.model_id,
            text_key,
            limit,
            max_length,
            medium_threshold,
            hard_threshold,
        )

        model.eval()
        row_iterator = self.iter_rows(dataset, limit=limit)
        if show_progress:
            row_iterator = tqdm(row_iterator, total=limit, desc=progress_desc)

        for index, row in enumerate(row_iterator):
            if text_key not in row:
                self.logger.debug("Skipping row index=%s missing key=%s", index, text_key)
                continue

            text = row[text_key]
            if not isinstance(text, str) or not text.strip():
                self.logger.debug("Skipping row index=%s empty/non-string text", index)
                continue

            perplexity = self._compute_perplexity_for_text(
                text=text,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            difficulty = self.difficulty_from_perplexity(
                perplexity,
                medium_threshold=medium_threshold,
                hard_threshold=hard_threshold,
            )

            # if index % max(1, log_every) == 0:
            #     self.logger.info(
            #         "Processed row index=%s perplexity=%.4f difficulty=%s score=%.3f",
            #         index,
            #         perplexity,
            #         difficulty["label"],
            #         difficulty["score"],
            #     )

            yield {
                "index": index,
                "text": text,
                "perplexity": perplexity,
                "difficulty_score": difficulty["score"],
                "difficulty_label": difficulty["label"],
                "record": row,
            }

        self.logger.info("Completed perplexity streaming run")

    def _compute_perplexity_for_text(
        self,
        *,
        text: str,
        model: Any,
        tokenizer: Any,
        max_length: int,
    ) -> float:
        """Compute perplexity for one text input."""
        torch = _import_torch_module()

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = _move_batch_to_model_device(encoded=encoded, model=model)
        labels = encoded["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**encoded, labels=labels)
            loss = outputs.loss
            ppl = torch.exp(loss).item()

        if math.isinf(ppl) or math.isnan(ppl):
            self.logger.warning("Perplexity is invalid for one record; returning inf")
            return float("inf")
        return float(ppl)

    def difficulty_from_perplexity(
        self,
        perplexity: float,
        *,
        medium_threshold: float = 20.0,
        hard_threshold: float = 60.0,
    ) -> dict[str, Any]:
        """Convert perplexity to a user-facing difficulty score and label."""
        if medium_threshold <= 0 or hard_threshold <= 0:
            raise ValueError("Difficulty thresholds must be positive values.")
        if hard_threshold <= medium_threshold:
            raise ValueError("hard_threshold must be greater than medium_threshold.")

        if math.isinf(perplexity) or math.isnan(perplexity):
            return {"score": 1.0, "label": "hard"}

        # Normalize using a capped linear scale up to the hard threshold.
        score = max(0.0, min(1.0, perplexity / hard_threshold))
        if perplexity < medium_threshold:
            label = "easy"
        elif perplexity < hard_threshold:
            label = "medium"
        else:
            label = "hard"
        return {"score": score, "label": label}

    def load_qwen25_05b(
        self,
        *,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> LoadedModel:
        """Backward-compatible alias for loading Qwen2.5-0.5B."""
        return self.load_model(
            model_id=model_id,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def create_difficulty_dataset(
        self,
        dataset: Any,
        *,
        loaded_model: LoadedModel,
        text_key: str = "text",
        limit: int | None = None,
        max_length: int = 512,
        show_progress: bool = True,
        progress_desc: str = "Computing perplexity",
        log_every: int = 100,
        medium_threshold: float = 20.0,
        hard_threshold: float = 60.0,
        unload_source_dataset: bool = True,
        unload_model_after: bool = False,
    ) -> Any:
        """Build a new dataset with perplexity and difficulty classes.

        Returns a Hugging Face ``datasets.Dataset`` with added columns:
        ``perplexity``, ``difficulty_score``, and ``difficulty_label``.
        """
        datasets = _import_datasets_module()
        self.logger.info(
            "Creating difficulty dataset text_key=%s limit=%s unload_source_dataset=%s unload_model_after=%s",
            text_key,
            limit,
            unload_source_dataset,
            unload_model_after,
        )

        records: list[dict[str, Any]] = []
        class_counts = {"easy": 0, "medium": 0, "hard": 0}

        for result in self.stream_perplexities(
            dataset,
            loaded_model=loaded_model,
            text_key=text_key,
            limit=limit,
            max_length=max_length,
            show_progress=show_progress,
            progress_desc=progress_desc,
            log_every=log_every,
            medium_threshold=medium_threshold,
            hard_threshold=hard_threshold,
        ):
            new_row = dict(result["record"])
            new_row["perplexity"] = result["perplexity"]
            new_row["difficulty_score"] = result["difficulty_score"]
            new_row["difficulty_label"] = result["difficulty_label"]
            records.append(new_row)
            class_counts[result["difficulty_label"]] += 1

        curated = datasets.Dataset.from_list(records)
        self.logger.info(
            "Created curated dataset rows=%s easy=%s medium=%s hard=%s",
            len(records),
            class_counts["easy"],
            class_counts["medium"],
            class_counts["hard"],
        )

        if unload_source_dataset:
            self.unload_dataset(dataset)
        if unload_model_after:
            self.unload_model(loaded_model)

        return curated

    def unload_dataset(self, dataset: Any | None) -> None:
        """Release references to a source dataset and trigger garbage collection."""
        if dataset is None:
            return
        self.logger.info("Unloading source dataset from memory")
        del dataset
        gc.collect()

    def unload_model(self, loaded_model: LoadedModel | None) -> None:
        """Release model/tokenizer resources and clear device caches when available."""
        if loaded_model is None:
            return
        self.logger.info("Unloading model model_id=%s", loaded_model.model_id)

        model = loaded_model.model
        del loaded_model
        del model
        gc.collect()

        try:
            torch = _import_torch_module()
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cleared CUDA cache")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            self.logger.info("Cleared MPS cache")

    def split_dataset(
        self,
        dataset: Any,
        *,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by_difficulty: bool = True,
        seed: int = 42,
    ) -> Any:
        """Split curated dataset into train/val/test, optionally stratified by difficulty."""
        datasets = _import_datasets_module()
        total_ratio = train_ratio + val_ratio + test_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        total = len(dataset)
        if total == 0:
            raise ValueError("Cannot split an empty dataset.")

        self.logger.info(
            "Splitting dataset rows=%s train=%.2f val=%.2f test=%.2f stratified=%s",
            total,
            train_ratio,
            val_ratio,
            test_ratio,
            stratify_by_difficulty,
        )

        rng = random.Random(seed)
        train_count, val_count, test_count = _split_counts(
            total,
            train_ratio,
            val_ratio,
            test_ratio,
        )

        if not stratify_by_difficulty or "difficulty_label" not in dataset.column_names:
            indices = list(range(total))
            rng.shuffle(indices)
            train_indices = indices[:train_count]
            val_indices = indices[train_count : train_count + val_count]
            test_indices = indices[train_count + val_count :]
        else:
            buckets = self._build_class_indices(dataset, label_key="difficulty_label")
            train_indices: list[int] = []
            val_indices: list[int] = []
            test_indices: list[int] = []
            for label in ["easy", "medium", "hard"]:
                label_indices = list(buckets.get(label, []))
                rng.shuffle(label_indices)
                n = len(label_indices)
                l_train, l_val, _ = _split_counts(n, train_ratio, val_ratio, test_ratio)
                train_indices.extend(label_indices[:l_train])
                val_indices.extend(label_indices[l_train : l_train + l_val])
                test_indices.extend(label_indices[l_train + l_val :])
            rng.shuffle(train_indices)
            rng.shuffle(val_indices)
            rng.shuffle(test_indices)

        split = datasets.DatasetDict(
            {
                "train": dataset.select(train_indices),
                "validation": dataset.select(val_indices),
                "test": dataset.select(test_indices),
            }
        )
        self.logger.info(
            "Split complete train=%s validation=%s test=%s",
            len(split["train"]),
            len(split["validation"]),
            len(split["test"]),
        )
        return split

    def create_curriculum_dataloader(
        self,
        dataset: Any,
        *,
        tokenizer: Any,
        text_key: str = "text",
        total_epochs: int = 5,
        steps_per_epoch: int = 100,
        batch_size: int = 2,
        max_length: int = 128,
        seed: int = 42,
        schedule: CurriculumSchedule | None = None,
        num_workers: int = 0,
    ) -> Any:
        """Create a dynamic curriculum DataLoader from a curated dataset split."""
        torch = _import_torch_module()
        if "difficulty_label" not in dataset.column_names:
            raise ValueError(
                "Dataset must include 'difficulty_label'. "
                "Build it first with create_difficulty_dataset()."
            )
        if text_key not in dataset.column_names:
            raise ValueError(f"Dataset is missing text column '{text_key}'.")

        class_indices = self._build_class_indices(dataset, label_key="difficulty_label")
        total_steps = max(1, total_epochs * steps_per_epoch)
        use_schedule = schedule or CurriculumSchedule.default()

        self.logger.info(
            "Creating curriculum dataloader epochs=%s steps_per_epoch=%s total_steps=%s batch_size=%s",
            total_epochs,
            steps_per_epoch,
            total_steps,
            batch_size,
        )
        self.logger.info(
            "Class bucket sizes easy=%s medium=%s hard=%s",
            len(class_indices["easy"]),
            len(class_indices["medium"]),
            len(class_indices["hard"]),
        )

        iterable = CurriculumIterableDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            class_indices=class_indices,
            text_key=text_key,
            total_steps=total_steps,
            max_length=max_length,
            seed=seed,
            schedule=use_schedule,
        )
        return torch.utils.data.DataLoader(
            iterable,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate_curriculum_batch,
        )

    def _build_class_indices(self, dataset: Any, *, label_key: str) -> dict[str, list[int]]:
        buckets = {"easy": [], "medium": [], "hard": []}
        for idx, label in enumerate(dataset[label_key]):
            if label in buckets:
                buckets[label].append(idx)
        return buckets


def _import_datasets_module() -> Any:
    """Import datasets lazily to keep import-time overhead low."""
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with `pip install datasets`."
        ) from exc
    return datasets


def _import_torch_module() -> Any:
    """Import torch lazily for model loading APIs."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The 'torch' package is required for model loading. "
            "Install with `pip install torch`."
        ) from exc
    return torch


def _import_transformers_module() -> Any:
    """Import transformers lazily for model loading APIs."""
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for model loading. "
            "Install with `pip install transformers`."
        ) from exc
    return transformers


def _has_bitsandbytes() -> bool:
    """Check if bitsandbytes is available for 4-bit/8-bit quantization."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def _import_tqdm() -> Any:
    """Import tqdm with a clear dependency message."""
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "The 'tqdm' package is required for progress bars. "
            "Install with `pip install tqdm`."
        ) from exc
    return tqdm


def _move_batch_to_model_device(*, encoded: dict[str, Any], model: Any) -> dict[str, Any]:
    """Move tokenized inputs to an appropriate model device when possible."""
    # If model is sharded (device_map='auto'), accelerate handles placement.
    if hasattr(model, "hf_device_map"):
        return encoded

    model_device = getattr(model, "device", None)
    if model_device is None:
        return encoded

    moved: dict[str, Any] = {}
    for key, value in encoded.items():
        if hasattr(value, "to"):
            moved[key] = value.to(model_device)
        else:
            moved[key] = value
    return moved


def _normalize_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    total = weights[0] + weights[1] + weights[2]
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def _split_counts(
    total: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if test_count < 0:
        test_count = 0
    return train_count, val_count, test_count


def _collate_curriculum_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _import_torch_module()
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "difficulty_label": [item["difficulty_label"] for item in batch],
        "perplexity": [item["perplexity"] for item in batch],
    }
