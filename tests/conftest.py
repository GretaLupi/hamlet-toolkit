"""Shared fixtures.

Kept minimal: the only thing here is workspace isolation, which every test that
touches the interface needs and none of them can arrange for itself.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_workspace(tmp_path_factory):
    """Keep the suite out of the user's real results directory.

    The interface writes projects, screenings, analyses and uploads under
    `results/` in a source checkout. Tests that go through the HTTP routes
    cannot be handed a workspace argument -- the routes take a form, not a
    destination -- so without this they leave real directories behind in the
    tree, and a stale one then shows up in the interface's own list of models
    the user trained.
    """
    root = tmp_path_factory.mktemp("hamlet-workspace")
    previous = os.environ.get("HAMLET_WORKSPACE")
    os.environ["HAMLET_WORKSPACE"] = str(root)
    yield root
    if previous is None:
        os.environ.pop("HAMLET_WORKSPACE", None)
    else:
        os.environ["HAMLET_WORKSPACE"] = previous
