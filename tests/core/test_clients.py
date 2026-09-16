"""Tests for core.clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ragdoll.core import clients


def test_embed_texts_returns_embeddings_and_latency(mocker: MockerFixture) -> None:
    fake_model = mocker.Mock()
    fake_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mocker.patch.object(clients, 'SentenceTransformer', return_value=fake_model)

    embeddings, latency_ms = clients.embed_texts(['a', 'b'], 'fake-model')

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert latency_ms >= 0
    fake_model.encode.assert_called_once_with(['a', 'b'])


def test_embed_texts_caches_model_per_name(mocker: MockerFixture) -> None:
    mock_cls = mocker.patch.object(clients, 'SentenceTransformer')
    mock_cls.return_value.encode.return_value = []

    clients.embed_texts(['a'], 'same-model')
    clients.embed_texts(['b'], 'same-model')

    mock_cls.assert_called_once_with('same-model', device='cpu')


def test_embed_texts_reloads_model_for_a_different_device(mocker: MockerFixture) -> None:
    mock_cls = mocker.patch.object(clients, 'SentenceTransformer')
    mock_cls.return_value.encode.return_value = []

    clients.embed_texts(['a'], 'same-model', device='cpu')
    clients.embed_texts(['b'], 'same-model', device='cuda:0')

    mock_cls.assert_any_call('same-model', device='cpu')
    mock_cls.assert_any_call('same-model', device='cuda:0')


def test_generate_chat_returns_answer_latency_and_token_usage(mocker: MockerFixture) -> None:
    fake_response = mocker.Mock()
    fake_response.choices = [mocker.Mock(message=mocker.Mock(content='the answer'))]
    fake_response.usage = mocker.Mock(prompt_tokens=10, completion_tokens=5)
    fake_client = mocker.Mock()
    fake_client.chat.completions.create.return_value = fake_response
    mocker.patch.object(clients, 'OpenAI', return_value=fake_client)

    answer, latency_ms, token_usage = clients.generate_chat(
        [{'role': 'user', 'content': 'hi'}],
        'fake-model',
        max_tokens=100,
        temperature=0.0,
    )

    assert answer == 'the answer'
    assert latency_ms >= 0
    assert token_usage == {'prompt_tokens': 10, 'completion_tokens': 5}
