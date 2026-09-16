"""Expands a combination grid, builds/reuses indexes, and runs pipelines."""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragdoll import LOG_FORMAT
from ragdoll.core.metrics.evaluate import EvaluationResult, evaluate_combination
from ragdoll.core.metrics.faithfulness import DEFAULT_JUDGE_MODEL
from ragdoll.core.pipeline import StageCombination, build_pipeline_stages, run_pipeline_stages
from ragdoll.core.registry import chunkers, incompatibility_reason, is_compatible, retrievers

if TYPE_CHECKING:
    from ragdoll.core.schema import Document, IndexHandle, Query, RAGResponse

logger = logging.getLogger(__name__)


class IncompatibleStagesError(ValueError):
    """Raised when build_index is called with a declared-incompatible chunker/retriever pair."""

    def __init__(self, chunker_name: str, retriever_name: str, reason: str | None) -> None:
        super().__init__(f"'{chunker_name}' is incompatible with '{retriever_name}': {reason}")


def expand_grid(
    chunkers: list[str],
    query_transforms: list[str],
    retrievers: list[str],
    rerankers: list[str],
    generators: list[str],
) -> list[StageCombination]:
    """Enumerate every stage combination across the given per-slot options.

    Combinations whose (chunker, retriever) pair is declared incompatible in
    the registry are excluded rather than raising — see AGENTS.md.
    """
    return [
        StageCombination(chunker=c, query_transform=qt, retriever=r, reranker=rr, generator=g)
        for c, qt, r, rr, g in product(chunkers, query_transforms, retrievers, rerankers, generators)
        if is_compatible(c, r)
    ]


def build_index(
    chunker: str,
    retriever: str,
    documents: list[Document],
    chunker_config: dict[str, Any] | None = None,
    retriever_config: dict[str, Any] | None = None,
) -> IndexHandle:
    """Build and return the index handle for one (chunker, retriever) pair."""
    if not is_compatible(chunker, retriever):
        raise IncompatibleStagesError(chunker, retriever, incompatibility_reason(chunker, retriever))

    chunker_entry = chunkers.get(chunker)
    chunker_instance = chunker_entry.cls(chunker_entry.config_model(**(chunker_config or {})))
    chunks = chunker_instance(documents)

    retriever_entry = retrievers.get(retriever)
    retriever_instance = retriever_entry.cls(retriever_entry.config_model(**(retriever_config or {})))
    return retriever_instance.build_index(chunks)


def run_combination(
    combination: StageCombination,
    index_handle: IndexHandle,
    queries: list[Query],
) -> list[RAGResponse]:
    """Run every query through one stage combination against a built index."""
    stages = build_pipeline_stages(combination)
    return [run_pipeline_stages(stages, index_handle, query) for query in queries]


def _attach_run_file_handler(run_dir: Path) -> logging.Handler:
    """Add a file handler scoped to this run, on top of the package's pre-configured console handler."""
    handler = logging.FileHandler(run_dir / 'run.log')
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger('ragdoll').addHandler(handler)
    return handler


def _apply_configs(
    combination: StageCombination,
    *,
    chunker_configs: dict[str, dict[str, Any]] | None,
    query_transform_configs: dict[str, dict[str, Any]] | None,
    retriever_configs: dict[str, dict[str, Any]] | None,
    reranker_configs: dict[str, dict[str, Any]] | None,
    generator_configs: dict[str, dict[str, Any]] | None,
) -> StageCombination:
    """Attach each stage's declared config (looked up by that stage's name) to a grid-expanded combination."""
    return combination.model_copy(
        update={
            'chunker_config': (chunker_configs or {}).get(combination.chunker, {}),
            'query_transform_config': (query_transform_configs or {}).get(combination.query_transform, {}),
            'retriever_config': (retriever_configs or {}).get(combination.retriever, {}),
            'reranker_config': (reranker_configs or {}).get(combination.reranker, {}),
            'generator_config': (generator_configs or {}).get(combination.generator, {}),
        }
    )


def _index_key(combination: StageCombination) -> tuple[str, str, str, str]:
    return (
        combination.chunker,
        json.dumps(combination.chunker_config, sort_keys=True),
        combination.retriever,
        json.dumps(combination.retriever_config, sort_keys=True),
    )


def _write_result(run_dir: Path, index: int, result: EvaluationResult) -> None:
    c = result.combination
    name = f'{index:03d}_{c.chunker}-{c.query_transform}-{c.retriever}-{c.reranker}-{c.generator}.json'
    (run_dir / name).write_text(result.model_dump_json(indent=2))


def _write_summary(run_dir: Path, results: list[EvaluationResult]) -> None:
    with (run_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'chunker',
                'query_transform',
                'retriever',
                'reranker',
                'generator',
                'n_queries',
                'recall_at_k',
                'precision_at_k',
                'mrr',
                'ndcg_at_k',
                'exact_match',
                'f1',
                'faithfulness',
                'latency_p50_ms',
                'latency_p95_ms',
            ]
        )
        for r in results:
            c = r.combination
            writer.writerow(
                [
                    c.chunker,
                    c.query_transform,
                    c.retriever,
                    c.reranker,
                    c.generator,
                    r.n_queries,
                    r.retrieval.recall_at_k,
                    r.retrieval.precision_at_k,
                    r.retrieval.mrr,
                    r.retrieval.ndcg_at_k,
                    r.correctness.exact_match,
                    r.correctness.f1,
                    r.faithfulness.faithfulness,
                    r.efficiency.latency_p50_ms,
                    r.efficiency.latency_p95_ms,
                ]
            )


def run_experiment(
    documents: list[Document],
    queries: list[Query],
    *,
    chunkers: list[str],
    query_transforms: list[str],
    retrievers: list[str],
    rerankers: list[str],
    generators: list[str],
    chunker_configs: dict[str, dict[str, Any]] | None = None,
    query_transform_configs: dict[str, dict[str, Any]] | None = None,
    retriever_configs: dict[str, dict[str, Any]] | None = None,
    reranker_configs: dict[str, dict[str, Any]] | None = None,
    generator_configs: dict[str, dict[str, Any]] | None = None,
    reports_dir: Path | str = 'reports',
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> list[EvaluationResult]:
    """Run every combination in the grid, evaluate it, and write results under a fresh per-run folder.

    Each call gets its own `reports_dir/<timestamp>/` folder, so results from
    different runs never overwrite each other. Distinct (chunker, retriever,
    their configs) pairs are indexed once and reused across every combination
    that shares them, per AGENTS.md.

    A `*_configs` argument maps a stage name to the config dict for that name
    (e.g. `chunker_configs={'fixed_size': {'chunk_size': 500}}`), applied to
    every combination that uses that name. Omitted names get that stage's
    default config.
    """
    run_dir = Path(reports_dir) / datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    file_handler = _attach_run_file_handler(run_dir)

    try:
        combinations = [
            _apply_configs(
                c,
                chunker_configs=chunker_configs,
                query_transform_configs=query_transform_configs,
                retriever_configs=retriever_configs,
                reranker_configs=reranker_configs,
                generator_configs=generator_configs,
            )
            for c in expand_grid(chunkers, query_transforms, retrievers, rerankers, generators)
        ]
        total = len(chunkers) * len(query_transforms) * len(retrievers) * len(rerankers) * len(generators)
        skipped = total - len(combinations)
        logger.info('expanded grid: %d combinations (%d skipped as incompatible)', len(combinations), skipped)

        index_cache: dict[tuple[str, str, str, str], IndexHandle] = {}
        results: list[EvaluationResult] = []

        for i, combination in enumerate(combinations):
            key = _index_key(combination)
            if key not in index_cache:
                logger.info('building index: chunker=%s retriever=%s', combination.chunker, combination.retriever)
                start = time.perf_counter()
                index_cache[key] = build_index(
                    combination.chunker,
                    combination.retriever,
                    documents,
                    combination.chunker_config,
                    combination.retriever_config,
                )
                logger.info('index built in %.1fms', (time.perf_counter() - start) * 1000)

            logger.info('running combination %d/%d: %s', i + 1, len(combinations), combination.model_dump_json())
            responses = run_combination(combination, index_cache[key], queries)

            result = evaluate_combination(combination, responses, queries, judge_model=judge_model)
            logger.info(
                'evaluated: recall@k=%.3f f1=%.3f faithfulness=%.3f',
                result.retrieval.recall_at_k,
                result.correctness.f1,
                result.faithfulness.faithfulness,
            )
            results.append(result)
            _write_result(run_dir, i, result)

        _write_summary(run_dir, results)
        logger.info('run complete: %d result(s) written to %s', len(results), run_dir)
    finally:
        logging.getLogger('ragdoll').removeHandler(file_handler)
        file_handler.close()

    return results
