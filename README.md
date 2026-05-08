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
