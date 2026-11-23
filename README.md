# llm-distill

Implementations and experiments for knowledge-distillation research papers,
from Hinton et al. (2015) through on-policy LLM distillation. The papers live in
[`refs/`](refs/README.md); see [`refs/reading_guide.md`](refs/reading_guide.md)
for a suggested reading order.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # core (torch, numpy) + dev tools
uv sync --extra hf      # also transformers / datasets / accelerate
uv run pytest           # run tests
```

## Layout

```
src/llm_distill/   library code (losses, trainers, utilities)
tests/             pytest suite
experiments/       paper reproductions and exploratory scripts
refs/              research papers (PDFs) and reading notes
```
