"""Core module API for DataCurator."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import gc
import logging
import math
from pathlib import Path
import random
import time
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
        batch_size: int | None = None,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        oom_backoff_factor: float = 0.5,
        tune_for_throughput: bool = True,
        tuning_interval_chunks: int = 5,
        probe_scale_up: float = 1.2,
        probe_scale_down: float = 0.85,
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
        tqdm = _import_tqdm()

        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        current_batch_size = (
            batch_size
            if batch_size is not None
            else self._default_perplexity_batch_size(
                loaded_model.device, max_batch_size=max_batch_size
            )
        )
        current_batch_size = max(min_batch_size, min(max_batch_size, current_batch_size))
        chunk_counter = 0
        last_throughput: float | None = None

        self.logger.info(
            "Starting perplexity stream model_id=%s text_key=%s limit=%s max_length=%s medium_threshold=%.3f hard_threshold=%.3f batch_size=%s min_batch_size=%s max_batch_size=%s tune_for_throughput=%s",
            loaded_model.model_id,
            text_key,
            limit,
            max_length,
            medium_threshold,
            hard_threshold,
            current_batch_size,
            min_batch_size,
            max_batch_size,
            tune_for_throughput,
        )

        model.eval()
        row_iterator = self.iter_rows(dataset, limit=limit)
        if show_progress:
            row_iterator = tqdm(row_iterator, total=limit, desc=progress_desc)

        pending: list[tuple[int, dict[str, Any], str]] = []
        processed = 0
        for index, row in enumerate(row_iterator):
            if text_key not in row:
                self.logger.debug("Skipping row index=%s missing key=%s", index, text_key)
                continue

            text = row[text_key]
            if not isinstance(text, str) or not text.strip():
                self.logger.debug("Skipping row index=%s empty/non-string text", index)
                continue

            pending.append((index, row, text))
            if len(pending) < current_batch_size:
                continue
            old_batch_size = current_batch_size
            started = time.perf_counter()
            perplexities, current_batch_size = self._compute_perplexity_batch_with_backoff(
                texts=[item[2] for item in pending],
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                initial_batch_size=current_batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                oom_backoff_factor=oom_backoff_factor,
            )
            elapsed = max(1e-6, time.perf_counter() - started)
            chunk_counter += 1
            if tune_for_throughput:
                current_throughput = len(perplexities) / elapsed
                if last_throughput is None:
                    last_throughput = current_throughput
                elif chunk_counter % max(1, tuning_interval_chunks) == 0:
                    current_batch_size = self._tune_batch_size_for_throughput(
                        current_batch_size=current_batch_size,
                        current_throughput=current_throughput,
                        previous_throughput=last_throughput,
                        min_batch_size=min_batch_size,
                        max_batch_size=max_batch_size,
                        probe_scale_up=probe_scale_up,
                        probe_scale_down=probe_scale_down,
                    )
                    last_throughput = current_throughput

            if current_batch_size < old_batch_size:
                last_throughput = None
            for (p_index, p_row, p_text), perplexity in zip(pending, perplexities):
                difficulty = self.difficulty_from_perplexity(
                    perplexity,
                    medium_threshold=medium_threshold,
                    hard_threshold=hard_threshold,
                )
                processed += 1
                if processed % max(1, log_every) == 0:
                    self.logger.info("Processed rows=%s", processed)
                yield {
                    "index": p_index,
                    "text": p_text,
                    "perplexity": perplexity,
                    "difficulty_score": difficulty["score"],
                    "difficulty_label": difficulty["label"],
                    "record": p_row,
                }
            pending = []

        if pending:
            perplexities, current_batch_size = self._compute_perplexity_batch_with_backoff(
                texts=[item[2] for item in pending],
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                initial_batch_size=current_batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                oom_backoff_factor=oom_backoff_factor,
            )
            for (p_index, p_row, p_text), perplexity in zip(pending, perplexities):
                difficulty = self.difficulty_from_perplexity(
                    perplexity,
                    medium_threshold=medium_threshold,
                    hard_threshold=hard_threshold,
                )
                processed += 1
                yield {
                    "index": p_index,
                    "text": p_text,
                    "perplexity": perplexity,
                    "difficulty_score": difficulty["score"],
                    "difficulty_label": difficulty["label"],
                    "record": p_row,
                }

        self.logger.info("Completed perplexity streaming run")

    def _compute_perplexity_batch_with_backoff(
        self,
        *,
        texts: list[str],
        model: Any,
        tokenizer: Any,
        max_length: int,
        initial_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
        oom_backoff_factor: float,
    ) -> tuple[list[float], int]:
        """Compute perplexities for texts with adaptive memory-aware micro-batching."""
        torch = _import_torch_module()
        results: list[float] = []
        current_batch_size = max(min_batch_size, min(max_batch_size, initial_batch_size))
        position = 0

        while position < len(texts):
            end = min(position + current_batch_size, len(texts))
            chunk = texts[position:end]
            try:
                chunk_results = self._compute_perplexity_batch(
                    texts=chunk,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
                results.extend(chunk_results)
                position = end
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                if current_batch_size <= min_batch_size:
                    raise RuntimeError(
                        "Perplexity batch failed even at minimum batch size. "
                        "Reduce max_length or use a smaller scoring model."
                    ) from exc
                next_batch_size = max(
                    min_batch_size, int(current_batch_size * oom_backoff_factor)
                )
                if next_batch_size >= current_batch_size:
                    next_batch_size = current_batch_size - 1
                self.logger.warning(
                    "OOM during perplexity scoring; reducing batch_size from %s to %s",
                    current_batch_size,
                    next_batch_size,
                )
                current_batch_size = max(min_batch_size, next_batch_size)
                _clear_torch_cache(torch=torch)

        return results, current_batch_size

    def _compute_perplexity_batch(
        self,
        *,
        texts: list[str],
        model: Any,
        tokenizer: Any,
        max_length: int,
    ) -> list[float]:
        """Compute per-record perplexities for a batch of texts."""
        torch = _import_torch_module()
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = _move_batch_to_model_device(encoded=encoded, model=model)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Next-token prediction labels for causal LM.
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, :-1, :].contiguous()
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            neg_log_likelihood = -(token_log_probs * shift_mask).sum(dim=1)
            token_counts = shift_mask.sum(dim=1).clamp(min=1)
            avg_nll = neg_log_likelihood / token_counts
            perplexities = torch.exp(avg_nll).detach().float().cpu().tolist()

        normalized: list[float] = []
        for ppl in perplexities:
            if math.isinf(ppl) or math.isnan(ppl):
                normalized.append(float("inf"))
            else:
                normalized.append(float(ppl))
        return normalized

    def _default_perplexity_batch_size(
        self,
        device: str,
        *,
        max_batch_size: int,
    ) -> int:
        """Pick a conservative default batch size by device."""
        if device == "cuda":
            return min(max_batch_size, 32)
        if device == "mps":
            return min(max_batch_size, 16)
        return min(max_batch_size, 4)

    def _tune_batch_size_for_throughput(
        self,
        *,
        current_batch_size: int,
        current_throughput: float,
        previous_throughput: float,
        min_batch_size: int,
        max_batch_size: int,
        probe_scale_up: float,
        probe_scale_down: float,
    ) -> int:
        """Tune batch size to maximize records/sec instead of raw size."""
        if current_throughput >= previous_throughput:
            next_batch_size = max(
                current_batch_size + 1,
                int(current_batch_size * max(1.01, probe_scale_up)),
            )
            return min(max_batch_size, next_batch_size)

        next_batch_size = int(current_batch_size * min(0.99, probe_scale_down))
        return max(min_batch_size, min(current_batch_size - 1, next_batch_size))

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
        perplexity_batch_size: int | None = None,
        min_perplexity_batch_size: int = 1,
        max_perplexity_batch_size: int = 32,
        oom_backoff_factor: float = 0.5,
        tune_perplexity_for_throughput: bool = True,
        perplexity_tuning_interval_chunks: int = 1,
        perplexity_probe_scale_up: float = 1.2,
        perplexity_probe_scale_down: float = 0.85,
        spill_to_disk: bool = False,
        spill_dir: str | None = None,
        spill_chunk_size: int = 1000,
        unload_source_dataset: bool = True,
        unload_model_after: bool = False,
        reuse_cached_curated: bool = False,
        cached_curated_path: str | Path | None = None,
        reuse_spill_shards: bool = False,
        spill_shards_dir: str | Path | None = None,
    ) -> Any:
        """Build a new dataset with perplexity and difficulty classes.

        Returns a Hugging Face ``datasets.Dataset`` with added columns:
        ``perplexity``, ``difficulty_score``, and ``difficulty_label``.

        If ``reuse_cached_curated`` is True, loads a dataset previously saved with
        ``Dataset.save_to_disk`` from ``cached_curated_path`` and skips scoring.

        If ``reuse_spill_shards`` is True, loads ``shard_*`` subdirectories under
        ``spill_shards_dir`` (same layout as ``spill_to_disk``) and skips scoring.

        Defaults preserve existing behavior (always score from ``dataset``).
        """
        datasets = _import_datasets_module()
        if reuse_cached_curated and reuse_spill_shards:
            raise ValueError(
                "Use only one of reuse_cached_curated or reuse_spill_shards, not both."
            )
        if reuse_spill_shards:
            if spill_shards_dir is None:
                raise ValueError(
                    "spill_shards_dir is required when reuse_spill_shards=True."
                )
            self.logger.info("Reusing spill shards from %s", spill_shards_dir)
            curated = self.load_curated_from_spill_dir(spill_shards_dir)
            if unload_source_dataset:
                self.unload_dataset(dataset)
            if unload_model_after:
                self.unload_model(loaded_model)
            return curated

        if reuse_cached_curated:
            if cached_curated_path is None:
                raise ValueError(
                    "cached_curated_path is required when reuse_cached_curated=True."
                )
            cache_path = Path(cached_curated_path)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"No curated dataset at {cache_path}. "
                    "Save with Dataset.save_to_disk(...) or set reuse_cached_curated=False."
                )
            self.logger.info("Loading cached curated dataset from %s", cache_path)
            curated = datasets.load_from_disk(str(cache_path))
            if unload_source_dataset:
                self.unload_dataset(dataset)
            if unload_model_after:
                self.unload_model(loaded_model)
            return curated

        self.logger.info(
            "Creating difficulty dataset text_key=%s limit=%s spill_to_disk=%s unload_source_dataset=%s unload_model_after=%s",
            text_key,
            limit,
            spill_to_disk,
            unload_source_dataset,
            unload_model_after,
        )

        records: list[dict[str, Any]] = []
        class_counts = {"easy": 0, "medium": 0, "hard": 0}
        shard_paths: list[Path] = []
        if spill_to_disk:
            if spill_chunk_size <= 0:
                raise ValueError("spill_chunk_size must be > 0 when spill_to_disk=True.")
            base_dir = Path(spill_dir or "outputs/curated_spill")
            base_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Spilling curated shards to disk at %s", base_dir)
        else:
            base_dir = None

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
            batch_size=perplexity_batch_size,
            min_batch_size=min_perplexity_batch_size,
            max_batch_size=max_perplexity_batch_size,
            oom_backoff_factor=oom_backoff_factor,
            tune_for_throughput=tune_perplexity_for_throughput,
            tuning_interval_chunks=perplexity_tuning_interval_chunks,
            probe_scale_up=perplexity_probe_scale_up,
            probe_scale_down=perplexity_probe_scale_down,
        ):
            new_row = dict(result["record"])
            new_row["perplexity"] = result["perplexity"]
            new_row["difficulty_score"] = result["difficulty_score"]
            new_row["difficulty_label"] = result["difficulty_label"]
            records.append(new_row)
            class_counts[result["difficulty_label"]] += 1
            if spill_to_disk and len(records) >= spill_chunk_size:
                shard_path = self._write_curated_shard(base_dir=base_dir, records=records, shard_index=len(shard_paths))
                shard_paths.append(shard_path)
                records.clear()
                gc.collect()

        if spill_to_disk:
            if records:
                shard_path = self._write_curated_shard(base_dir=base_dir, records=records, shard_index=len(shard_paths))
                shard_paths.append(shard_path)
                records.clear()
            curated = self._load_curated_from_shards(datasets=datasets, shard_paths=shard_paths)
        else:
            curated = datasets.Dataset.from_list(records)
        self.logger.info(
            "Created curated dataset rows=%s easy=%s medium=%s hard=%s",
            len(curated),
            class_counts["easy"],
            class_counts["medium"],
            class_counts["hard"],
        )

        if unload_source_dataset:
            self.unload_dataset(dataset)
        if unload_model_after:
            self.unload_model(loaded_model)

        return curated

    def _write_curated_shard(
        self,
        *,
        base_dir: Path | None,
        records: list[dict[str, Any]],
        shard_index: int,
    ) -> Path:
        """Write one curated shard to disk and return the shard path."""
        if base_dir is None:
            raise ValueError("base_dir must be set for shard writing.")
        datasets = _import_datasets_module()
        shard_path = base_dir / f"shard_{shard_index:06d}"
        shard_dataset = datasets.Dataset.from_list(records)
        shard_dataset.save_to_disk(str(shard_path))
        return shard_path

    def _load_curated_from_shards(self, *, datasets: Any, shard_paths: list[Path]) -> Any:
        """Load all saved curated shards and concatenate into one dataset."""
        if not shard_paths:
            return datasets.Dataset.from_list([])
        shards = [datasets.load_from_disk(str(path)) for path in shard_paths]
        return datasets.concatenate_datasets(shards)

    def load_curated_from_spill_dir(self, spill_dir: str | Path) -> Any:
        """Load a curated dataset from a prior ``spill_to_disk`` directory.

        Expects child directories named ``shard_000000``, ``shard_000001``, …
        as produced when writing shards under ``spill_dir``.
        """
        datasets = _import_datasets_module()
        root = Path(spill_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Spill directory not found: {root}")
        shard_paths = sorted(
            p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_")
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_* subdirectories under {root}. "
                "Run curation with spill_to_disk=True first, or pass the spill root "
                "that contains shard_* folders."
            )
        self.logger.info("Loading %s spill shards from %s", len(shard_paths), root)
        return self._load_curated_from_shards(datasets=datasets, shard_paths=shard_paths)

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

    def build_curriculum_iterable_dataset(
        self,
        dataset: Any,
        *,
        tokenizer: Any,
        text_key: str = "text",
        total_epochs: int = 5,
        steps_per_epoch: int = 100,
        max_length: int = 128,
        seed: int = 42,
        schedule: CurriculumSchedule | None = None,
    ) -> CurriculumIterableDataset:
        """Build a :class:`CurriculumIterableDataset` for use with Hugging Face ``Trainer``.

        Each call to ``__iter__`` yields ``total_epochs * steps_per_epoch`` samples, drawing
        rows by difficulty with weights from ``schedule`` (default: easy → hard over progress).

        Pair with :func:`curriculum_lm_collate_fn` as ``data_collator`` and set
        ``TrainingArguments(max_steps=...)`` so one training run consumes the iterator
        (typically ``ceil((total_epochs * steps_per_epoch) / per_device_train_batch_size)``).
        """
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
            "Building curriculum iterable dataset epochs=%s steps_per_epoch=%s total_samples=%s max_length=%s",
            total_epochs,
            steps_per_epoch,
            total_steps,
            max_length,
        )
        self.logger.info(
            "Class bucket sizes easy=%s medium=%s hard=%s",
            len(class_indices["easy"]),
            len(class_indices["medium"]),
            len(class_indices["hard"]),
        )

        return CurriculumIterableDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            class_indices=class_indices,
            text_key=text_key,
            total_steps=total_steps,
            max_length=max_length,
            seed=seed,
            schedule=use_schedule,
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


def curriculum_lm_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate curriculum or map-style HF batches for causal LM training (e.g. ``Trainer``).

    Accepts ``input_ids`` / ``attention_mask`` as ``torch.Tensor`` (curriculum iterable)
    or as lists / NumPy arrays (typical ``datasets`` map batches). Sets ``labels`` to
    ``-100`` where ``attention_mask == 0`` so padding is ignored in the loss.
    """
    return _collate_curriculum_batch(batch)


def _rows_to_long_tensor(torch: Any, rows: list[Any]) -> Any:
    """Stack one row per batch element into a single ``long`` tensor."""
    tensors = []
    for row in rows:
        if isinstance(row, torch.Tensor):
            tensors.append(row.long())
        else:
            tensors.append(torch.as_tensor(row, dtype=torch.long))
    return torch.stack(tensors, dim=0)


def _collate_curriculum_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _import_torch_module()
    input_ids = _rows_to_long_tensor(torch, [item["input_ids"] for item in batch])
    attention_mask = _rows_to_long_tensor(
        torch, [item["attention_mask"] for item in batch]
    )
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _is_oom_error(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "oom" in message


def _clear_torch_cache(*, torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
