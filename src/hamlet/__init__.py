"""HamLeT — Hamiltonian Learning Toolkit."""

from .data import SpectroscopyDataset, generate_dataset
from .systems import (
    HomogeneousHeisenbergFamily,
    HomogeneousHeisenbergChain,
    InhomogeneousHeisenbergFamily,
    InhomogeneousHeisenbergChain,
    HomogeneousXXZLongRangeChain,
    HomogeneousXXZLongRangeFamily,
    HomogeneousXXZDMILongRangeChain,
    HomogeneousXXZDMILongRangeFamily,
)
from .project import (
    DatasetGenerationConfig,
    HamiltonianLearningProject,
    PlannedOutput,
    ProjectConfig,
    ProjectOutcome,
    ProjectPlan,
)
from .measurements import Measurement
from .experiments import (
    ExperimentInspectionResult,
    ExperimentModeSelectionResult,
    ExperimentModeProfile,
    available_experiment_modes,
    inspect_experiment_recipe,
    load_canonical_experiment,
    load_experiment_project,
    resolve_experiment_mode,
    select_experiment_mode,
)
from .workflow import ResourceAssessment, WorkflowDecision, advise_experiment

__all__ = [
    "HomogeneousHeisenbergChain",
    "HomogeneousHeisenbergFamily",
    "InhomogeneousHeisenbergChain",
    "HomogeneousXXZLongRangeChain",
    "HomogeneousXXZLongRangeFamily",
    "HomogeneousXXZDMILongRangeChain",
    "HomogeneousXXZDMILongRangeFamily",
    "InhomogeneousHeisenbergFamily",
    "HamiltonianLearningProject",
    "DatasetGenerationConfig",
    "PlannedOutput",
    "ProjectConfig",
    "ProjectPlan",
    "ProjectOutcome",
    "Measurement",
    "ExperimentInspectionResult",
    "ExperimentModeSelectionResult",
    "ExperimentModeProfile",
    "available_experiment_modes",
    "inspect_experiment_recipe",
    "load_canonical_experiment",
    "load_experiment_project",
    "resolve_experiment_mode",
    "select_experiment_mode",
    "ResourceAssessment",
    "WorkflowDecision",
    "advise_experiment",
    "SpectroscopyDataset",
    "generate_dataset",
]
__version__ = "0.1.0"
__brand__ = "HamLeT"
__full_name__ = "Hamiltonian Learning Toolkit"
