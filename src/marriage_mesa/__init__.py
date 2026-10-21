"""Mesa replication of the Grow & Van Bavel (2015) marriage-matching ABM."""

from .config import ModelConfig
from .inputs import CohortInputs
from .model import MarriageAgent, MarriageModel
from .workflow import WorkflowSpec, run_workflow

__all__ = [
    "ModelConfig", "CohortInputs", "MarriageAgent", "MarriageModel",
    "WorkflowSpec", "run_workflow",
]
