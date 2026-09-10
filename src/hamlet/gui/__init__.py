"""A local browser interface for the HamLeT workflows.

The command-line workflow assumes you already know which of several decisions
you are making. This exists because that assumption fails for most users,
including experienced ones returning to the package after a break: the hard
part is not typing the command, it is knowing *which* command the situation
calls for.

The interface is therefore organised around the questions rather than the
commands, and every screen states what it is about to do before doing it.

Start it with ``hamlet gui``. It binds to localhost only.
"""

from .api import (
    GuiJob,
    JobRegistry,
    advise_for_experiment,
    browse_directory,
    build_analysis_config,
    build_project_config,
    build_screening_config,
    describe_available_models,
    describe_builder_options,
    describe_output_locations,
    describe_screening_options,
    inspect_experiment,
    list_example_configs,
    plan_project,
    run_analysis,
    save_upload,
    screening_preview,
    workflow_overview,
)
from .server import serve

__all__ = [
    "GuiJob",
    "JobRegistry",
    "advise_for_experiment",
    "browse_directory",
    "build_analysis_config",
    "build_project_config",
    "build_screening_config",
    "describe_available_models",
    "describe_builder_options",
    "describe_output_locations",
    "describe_screening_options",
    "inspect_experiment",
    "list_example_configs",
    "plan_project",
    "run_analysis",
    "save_upload",
    "screening_preview",
    "serve",
    "workflow_overview",
]
