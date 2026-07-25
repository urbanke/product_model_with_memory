# product_model_with_memory

A simple product model but for sources with memory.

Companion project to `product_model`: same stack (Python with NumPy/SciPy),
same layout, extended to sources with memory.

## Environment

Create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The repository also includes a conda environment file:

```bash
conda env create -f environment.yml
conda activate product-model-with-memory
python -m pip install -e ".[dev]"
```

## Layout

- `src/product_model_with_memory/` — the package (importable modules)
- `tests/` — pytest test suite (`pytest` from the repo root)
- `scripts/` — experiment entry points
- `docs/` — notes and write-ups
- `output/` — generated results

## Checks

```bash
pytest
ruff check .
```
