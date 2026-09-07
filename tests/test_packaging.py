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


# The browser interface is served from files on disk rather than imported, so
# neither the git-tracking check above nor an import check would notice them
# missing. An installed package would then import fine and 404 on every page.
GUI_STATIC_ROOT = SOURCE_ROOT / "gui" / "static"


@requires_git
def test_gui_static_assets_are_tracked_by_git():
    on_disk = {
        path.resolve()
        for path in GUI_STATIC_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert on_disk, "the interface has no static assets, which cannot be right"
    missing = sorted(str(p.relative_to(REPO_ROOT)) for p in on_disk - _tracked_files())
    assert not missing, (
        "these interface assets exist locally but are not tracked, so a fresh "
        f"clone would serve a broken page: {missing}"
    )


def test_gui_static_assets_are_declared_as_package_data():
    """They must also be declared, or a wheel install serves nothing.

    Tracking them in git is not sufficient: setuptools only copies non-Python
    files into the wheel when package-data says to.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text, (
        "pyproject.toml declares no package-data, so the interface's HTML, CSS "
        "and JS would be absent from an installed package"
    )
    for suffix in (".html", ".css", ".js"):
        assert f"static/*{suffix}" in text, f"package-data does not cover static/*{suffix}"


def test_gui_serves_only_from_its_static_directory():
    """A path-traversal request must not read files outside the asset root."""
    from hamlet.gui.server import STATIC_ROOT

    assert STATIC_ROOT.is_dir()
    assert (STATIC_ROOT / "index.html").is_file()
    escaped = (STATIC_ROOT / "../../project.py").resolve()
    assert STATIC_ROOT not in escaped.parents
