"""Guards that the published repository actually contains the whole package.

A .gitignore rule naming a directory without a leading slash matches that name
at any depth. The root-level `data/` and `models/` rules therefore silently
excluded `src/hamlet/data/` and `src/hamlet/models/` from version control, so
`import hamlet` failed for anyone who cloned the repository even though every
test passed locally, where the files exist on disk.

These tests compare the working tree against what git tracks, which is the only
way to catch a file that is present locally but missing from a fresh clone.
"""

from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "hamlet"


def _tracked_files() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/hamlet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        (REPO_ROOT / item).resolve()
        for item in result.stdout.split("\0")
        if item
    }


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


requires_git = pytest.mark.skipif(
    not _git_available(), reason="not a git checkout (installed package or archive)"
)


@requires_git
def test_every_source_file_is_tracked_by_git():
    on_disk = {
        path.resolve()
        for path in SOURCE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(str(path.relative_to(REPO_ROOT)) for path in on_disk - _tracked_files())
    assert not missing, (
        "these source files exist locally but are not tracked, so a fresh clone "
        f"would be missing them: {missing}"
    )


@requires_git
def test_every_subpackage_is_tracked_by_git():
    tracked = _tracked_files()
    missing = []
    for init in SOURCE_ROOT.rglob("__init__.py"):
        if "__pycache__" in init.parts:
            continue
        if init.resolve() not in tracked:
            missing.append(str(init.parent.relative_to(REPO_ROOT)))
    assert not missing, f"untracked subpackages would break a fresh clone: {sorted(missing)}"


def test_declared_public_api_is_importable():
    """A missing subpackage breaks `import hamlet` at its first re-export."""
    import hamlet

    for name in hamlet.__all__:
        assert hasattr(hamlet, name), f"hamlet.__all__ advertises missing {name!r}"
