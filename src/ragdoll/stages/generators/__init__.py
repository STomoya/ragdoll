"""Generator implementations: (Query, list[RetrievedContext]) -> RAGResponse."""

from ragdoll.stages.generators.single_shot import SingleShotGenerator, SingleShotGeneratorConfig

__all__ = ['SingleShotGenerator', 'SingleShotGeneratorConfig']
