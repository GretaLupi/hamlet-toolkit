import hamlet

from hamlet.project_cli import build_parser
from hamlet.training import TrainingRun


def test_hamlet_brand_identity():
    assert hamlet.__brand__ == "HamLeT"
    assert hamlet.__full_name__ == "Hamiltonian Learning Toolkit"
    assert TrainingRun.__module__.startswith("hamlet")


def test_canonical_cli_name_is_hamlet():
    parser = build_parser()
    assert parser.prog == "hamlet"
    assert "Hamiltonian Learning Toolkit" in parser.description
