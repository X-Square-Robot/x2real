from manaenv.pipeline.artifacts.base import ArtifactLedger, BaseArtifact
from manaenv.pipeline.artifacts.evaluation import EvalArtifact

ArtifactLedger.register_type("eval", EvalArtifact)

__all__ = [
    "BaseArtifact",
    "ArtifactLedger",
    "EvalArtifact",
]
