# llm-distill

Implementations and experiments for knowledge-distillation research papers,
from Hinton et al. (2015) through on-policy LLM distillation. 

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # core (torch, numpy) + dev tools
uv sync --extra hf      # also transformers / datasets / accelerate
```

## Layout

```
src/llm_distill/   library code (losses, trainers, utilities)
experiments/       paper reproductions and exploratory scripts
```
