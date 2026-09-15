# ragdoll

Staged, method-agnostic RAG implementation and evaluation harness. See
[CLAUDE.md](CLAUDE.md) for the full architecture and data model.

## Install

```bash
uv sync
```

## Usage

```python
import ragdoll.stages  # registers every stage implementation

from ragdoll.core.pipeline import StageCombination, run_pipeline
from ragdoll.core.runner import build_index
from ragdoll.core.schema import Document, Query

documents = [Document(doc_id="d1", text="RAG combines retrieval with generation.")]
index_handle = build_index(chunker="fixed_size", retriever="dense", documents=documents)

combination = StageCombination(chunker="fixed_size", retriever="dense", generator="single_shot")
query = Query(query_id="q1", text="What is RAG?", lang="en")

response = run_pipeline(combination, index_handle, query)
print(response.answer)
```

## Development

```bash
uv run pytest    # tests
uvx ty check     # type check
uvx ruff check . # lint
```
