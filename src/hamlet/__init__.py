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
    HomogeneousXXZDMIFieldChain,
    HomogeneousXXZDMIFieldFamily,
    HomogeneousXXZDMIImpurityChain,
    HomogeneousXXZDMIImpurityFamily,
    SiteImpurity,
)
from .dmi_design import (
    DmiDesign,
    DmiImprint,
    format_screening_table,
    measure_dmi_imprint,
    screen_dmi_designs,
    transverse_impurities,
)
from .project import (
    DatasetGenerationConfig,
    HamiltonianLearningProject,
    PlannedOutput,
    ProjectConfig,
    ProjectOutcome,
    ProjectPlan,
    TuningConfig,
)
from .cancellation import CancelToken, OperationCancelled
from .cluster import (
    BUILT_IN_PROFILES,
    ClusterConfig,
    ClusterSession,
    SchedulerProfile,
    available_profiles,
    render_job_script,
)
from .compute import (
    ComputeReport,
    DeviceRequest,
    advise_device,
    configure_device,
    describe_compute,
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
    "HomogeneousXXZDMIFieldChain",
    "HomogeneousXXZDMIFieldFamily",
    "HomogeneousXXZDMIImpurityChain",
    "HomogeneousXXZDMIImpurityFamily",
    "SiteImpurity",
    "DmiDesign",
    "DmiImprint",
    "measure_dmi_imprint",
    "screen_dmi_designs",
    "transverse_impurities",
    "format_screening_table",
    "InhomogeneousHeisenbergFamily",
    "HamiltonianLearningProject",
    "DatasetGenerationConfig",
    "PlannedOutput",
    "ProjectConfig",
    "ProjectPlan",
    "ProjectOutcome",
    "TuningConfig",
    "BUILT_IN_PROFILES",
    "CancelToken",
    "ClusterConfig",
    "ClusterSession",
    "ComputeReport",
    "DeviceRequest",
    "OperationCancelled",
    "SchedulerProfile",
    "advise_device",
    "available_profiles",
    "configure_device",
    "describe_compute",
    "render_job_script",
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
