"""Natural Questions (KILT format) benchmark adapter.

facebook/kilt_wikipedia can't be loaded through the `datasets` library
(its script-based loader is no longer supported by current `datasets`
versions), so the knowledge source is streamed directly from the URL that
loader used to download.
"""

from __future__ import annotations

import json

import datasets
import requests

from ragdoll.core.schema import Document, Query

_TASKS_REPO = 'facebook/kilt_tasks'
_KNOWLEDGE_SOURCE_URL = 'http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json'


def load_natural_questions(
    split: str = 'validation',
    n_queries: int = 5,
    n_distractors: int = 20,
) -> tuple[list[Document], list[Query]]:
    """Load n_queries real NQ/KILT queries plus their gold documents and n_distractors distractors."""
    task_rows = datasets.load_dataset(_TASKS_REPO, 'nq', split=f'{split}[:{n_queries}]')

    queries: list[Query] = []
    gold_ids: set[str] = set()
    for row in task_rows:
        answers = [output['answer'] for output in row['output'] if output['answer']]
        doc_ids = {provenance['wikipedia_id'] for output in row['output'] for provenance in output['provenance']}
        gold_ids |= doc_ids
        queries.append(
            Query(
                query_id=row['id'],
                text=row['input'],
                lang='en',
                gold_answers=answers,
                gold_doc_ids=sorted(doc_ids),
            ),
        )

    documents = _collect_documents(gold_ids, n_distractors)
    return documents, queries


def _collect_documents(gold_ids: set[str], n_distractors: int) -> list[Document]:
    documents: list[Document] = []
    found_gold_ids: set[str] = set()
    distractor_count = 0

    response = requests.get(_KNOWLEDGE_SOURCE_URL, stream=True, timeout=30)
    try:
        # ponytail: linear scan with early exit -- fine for a handful of
        # queries; if gold ids are scattered this can still scan a large
        # prefix of the dump. A real subset needs an indexed lookup or a
        # pre-filtered local KILT dump.
        for line in response.iter_lines():
            if not line:
                continue
            article = json.loads(line)
            is_gold = article['wikipedia_id'] in gold_ids
            if not is_gold and distractor_count >= n_distractors:
                continue
            documents.append(
                Document(
                    doc_id=article['wikipedia_id'],
                    text=''.join(article['text']),
                    title=article['wikipedia_title'],
                ),
            )
            if is_gold:
                found_gold_ids.add(article['wikipedia_id'])
            else:
                distractor_count += 1
            if found_gold_ids == gold_ids and distractor_count >= n_distractors:
                break
    finally:
        response.close()
    return documents
