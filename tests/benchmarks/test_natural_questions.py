"""Tests for the Natural Questions (KILT) benchmark adapter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ragdoll.benchmarks.natural_questions import load_natural_questions

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _fake_task_rows() -> list[dict[str, Any]]:
    return [
        {
            'id': 'q1',
            'input': 'what is the therefore sign?',
            'output': [
                {'answer': 'the therefore sign', 'meta': {'score': -1}, 'provenance': [{'wikipedia_id': '10593264'}]},
                {'answer': '', 'meta': {'score': -1}, 'provenance': []},
            ],
        },
    ]


def _fake_wiki_lines() -> list[bytes]:
    lines = [
        {'wikipedia_id': '10593264', 'wikipedia_title': 'Therefore sign', 'text': ['Gold paragraph.\n']},
        {'wikipedia_id': '1', 'wikipedia_title': 'Distractor One', 'text': ['Distractor text.\n']},
        {'wikipedia_id': '2', 'wikipedia_title': 'Distractor Two', 'text': ['More distractor text.\n']},
    ]
    return [json.dumps(line).encode('utf-8') for line in lines]


def test_load_natural_questions_returns_queries_and_documents(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.benchmarks.natural_questions.datasets.load_dataset',
        return_value=_fake_task_rows(),
    )
    fake_response = mocker.Mock()
    fake_response.iter_lines.return_value = _fake_wiki_lines()
    mocker.patch('ragdoll.benchmarks.natural_questions.requests.get', return_value=fake_response)

    documents, queries = load_natural_questions(n_queries=1, n_distractors=2)

    assert len(queries) == 1
    assert queries[0].query_id == 'q1'
    assert queries[0].gold_answers == ['the therefore sign']
    assert queries[0].gold_doc_ids == ['10593264']

    doc_ids = {d.doc_id for d in documents}
    assert doc_ids == {'10593264', '1', '2'}
    gold_doc = next(d for d in documents if d.doc_id == '10593264')
    assert gold_doc.text == 'Gold paragraph.\n'
    assert gold_doc.title == 'Therefore sign'
    fake_response.close.assert_called_once()
