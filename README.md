# DataCurator

Install locally:

```bash
pip install -e .
```

Use as an imported module:

```python
from datacurator import DataCurator

curator = DataCurator()

# Regular loading (downloads/cache locally)
train = curator.load_dataset("ag_news", split="train")
print(train[0])

# Streaming (no full local download)
stream = curator.stream_dataset("HuggingFaceFW/fineweb", split="train")
for row in curator.iter_rows(stream, limit=3):
    print(row)
```

Load models efficiently (defaults to Qwen2.5-0.5B):

```python
from datacurator import DataCurator

curator = DataCurator()
loaded = curator.load_model()  # default: Qwen/Qwen2.5-0.5B-Instruct
print(loaded.model_id, loaded.device, loaded.quantization)

# Or choose any other model
loaded = curator.load_model(model_id="Qwen/Qwen2.5-1.5B-Instruct")
print(loaded.model_id, loaded.device, loaded.quantization)
```

Quantization behavior:
- CUDA + `bitsandbytes` installed: 4-bit NF4 quantization.
- CUDA without `bitsandbytes`: FP16 fallback.
- Apple Silicon (MPS): FP16 fallback.
- CPU: FP32 fallback.

Stream records and compute perplexity:

```python
from datacurator import DataCurator

curator = DataCurator()
dataset = curator.stream_dataset("wikitext", config="wikitext-2-raw-v1", split="test")
loaded = curator.load_model()  # defaults to Qwen/Qwen2.5-0.5B-Instruct

for result in curator.stream_perplexities(
    dataset,
    loaded_model=loaded,
    text_key="text",
    limit=100,
    show_progress=True,
):
    print(
        result["index"],
        result["perplexity"],
        result["difficulty_label"],
        result["difficulty_score"],
    )
```

Logging:
- DataCurator prints informative logs for dataset loading, model loading, and perplexity progress.
- Per-row progress is shown with `tqdm` when `show_progress=True`.
- Difficulty is computed from perplexity for each record:
  - `difficulty_label`: `easy`, `medium`, or `hard`
  - `difficulty_score`: normalized value in `[0.0, 1.0]`

Create a new curated dataset (and unload old resources):

```python
from datacurator import DataCurator

curator = DataCurator()
source = curator.stream_dataset("wikitext", config="wikitext-2-raw-v1", split="test")
loaded = curator.load_model()

curated = curator.create_difficulty_dataset(
    source,
    loaded_model=loaded,
    text_key="text",
    limit=100,
    show_progress=True,
    perplexity_batch_size=None,      # auto-select by device memory
    max_perplexity_batch_size=32,    # upper bound for adaptive batching
    tune_perplexity_for_throughput=True,   # optimize for records/sec
    spill_to_disk=True,                    # out-of-core mode for large datasets
    spill_dir="outputs/curated_spill",
    spill_chunk_size=500,
    unload_source_dataset=True,  # drop old dataset refs
    unload_model_after=True,     # optional model cleanup
)

print(curated[0]["difficulty_label"], curated[0]["perplexity"])
```

Reuse a curated dataset saved with Hugging Face ``save_to_disk`` (skips perplexity scoring):

```python
# After first run: curated.save_to_disk("outputs/my_curated")
curated = curator.create_difficulty_dataset(
    source,  # still passed; dropped if unload_source_dataset=True
    loaded_model=loaded,
    reuse_cached_curated=True,
    cached_curated_path="outputs/my_curated",
    unload_source_dataset=True,
    unload_model_after=True,
)
```

Reuse spill shards from an earlier ``spill_to_disk`` run (``shard_*`` folders under ``spill_dir``):

```python
curated = curator.load_curated_from_spill_dir("outputs/curated_spill")
```

You can also pass ``reuse_spill_shards=True`` and ``spill_shards_dir=...`` into ``create_difficulty_dataset`` if you prefer the same call site as a full curation run.

Perplexity speedup:
- Adaptive batch scoring is enabled by default and automatically backs off batch size on OOM.
- Batch size is tuned for throughput (records/sec), not just maximum size.
- Tune with `perplexity_batch_size`, `min_perplexity_batch_size`, `max_perplexity_batch_size`,
  `tune_perplexity_for_throughput`, `perplexity_tuning_interval_chunks`,
  `perplexity_probe_scale_up`, and `perplexity_probe_scale_down`.
- For large corpora, use out-of-core spilling: `spill_to_disk=True` with `spill_chunk_size`.

Split and curriculum sampling (library-managed):

```python
from datacurator import CurriculumSchedule, DataCurator
from transformers import GPT2TokenizerFast

curator = DataCurator()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# split curated dataset with stratification by difficulty labels
splits = curator.split_dataset(
    curated,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_by_difficulty=True,
)

# aggressive for few epochs, gradual for many epochs automatically
dataloader = curator.create_curriculum_dataloader(
    splits["train"],
    tokenizer=tokenizer,
    total_epochs=3,      # use 300 for gradual pacing
    steps_per_epoch=200,
    batch_size=4,
    max_length=128,
    schedule=CurriculumSchedule.default(),
)
```

## Experiments (curate then train)

Curation is heavy (loads the scorer model and scores every row). The experiment scripts split that work from training:

1. **Curate and save** (writes `outputs/curated_wikitext_train`):

   ```bash
   python experiments/curate_dataset.py
   ```

2. **Train on the saved dataset** (loads from disk, no scoring):

   ```bash
   python experiments/curated_training.py
   ```

3. **Optional — curriculum training** (same disk dataset; train split is sampled easy→hard over progress via ``CurriculumSchedule``):

   ```bash
   python experiments/curated_curriculum_training.py
   ```

The curated output path is shared by ``curate_dataset.py`` and the training scripts; change it in each file if you relocate the dataset.
