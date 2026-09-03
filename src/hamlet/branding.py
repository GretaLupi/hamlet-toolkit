"""Canonical HamLeT identity shared by CLIs, manifests, and reports."""

BRAND_NAME = "HamLeT"
FULL_NAME = "Hamiltonian Learning Toolkit"
DISTRIBUTION_NAME = "hamlet-toolkit"
CANONICAL_COMMAND = "hamlet"


def brand_manifest() -> dict[str, str]:
    return {
        "name": BRAND_NAME,
        "full_name": FULL_NAME,
        "distribution": DISTRIBUTION_NAME,
    }

