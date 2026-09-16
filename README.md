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

## Running a benchmark

Benchmark adapters produce `Document`/`Query` pairs against which you can run
any stage combination:

```python
import ragdoll.stages  # registers every stage implementation

from ragdoll.benchmarks.natural_questions import load_natural_questions
from ragdoll.core.pipeline import StageCombination
from ragdoll.core.runner import build_index, run_combination

documents, queries = load_natural_questions(n_queries=5)
index_handle = build_index(chunker="fixed_size", retriever="dense", documents=documents)

combination = StageCombination(chunker="fixed_size", retriever="dense", generator="single_shot")
responses = run_combination(combination, index_handle, queries)

for response in responses:
    print(response.query_id, response.answer)
```

Requires an `OPENAI_API_KEY` for the generator's LLM calls.

## Evaluating results

`evaluate_combination` scores a combination's responses against the
queries' gold fields (retrieval quality, answer correctness, efficiency,
and LLM-judged faithfulness), continuing from the benchmark example above:

```python
from ragdoll.core.metrics.evaluate import evaluate_combination

result = evaluate_combination(combination, responses, queries)
print(result.retrieval.recall_at_k, result.correctness.f1)
```

## Development

```bash
uv run pytest    # tests
uvx ty check     # type check
uvx ruff check . # lint
```
