# Releasing HamLeT

Releases are built by GitHub Actions and uploaded to PyPI with Trusted
Publishing. No PyPI API token is stored in the repository.

## One-time repository setup

1. Create the `hamlet-toolkit` project on PyPI, or create a pending Trusted
   Publisher with owner `GretaLupi`, repository `hamlet-toolkit`, workflow
   `release.yml`, and environment `pypi`.
2. In the GitHub repository settings, create an environment named `pypi`.
   Requiring manual approval for that environment is recommended.
3. Enable GitHub private vulnerability reporting in **Settings → Security**.

The PyPI project name must be registered before the first release. Every later
release uses the trusted publisher configured above.

## Release checklist

1. Choose the version and update it in `pyproject.toml`,
   `src/hamlet/__init__.py`, and `CITATION.cff`.
2. Move the relevant `CHANGELOG.md` entries from **Unreleased** into a dated
   version section.
3. Run:

   ```bash
   pytest
   ruff check src tests
   rm -rf dist
   python -m build
   python -m twine check dist/*
   ```

   `twine check` is what catches a README that PyPI will not render. Note
   that PyPI resolves relative links against `pypi.org`, so images and
   documentation links in `README.md` must stay absolute URLs.

4. Commit and push the release changes to `main`. There is no CI
   workflow, so step 3 is the gate: it has to pass locally.
5. Create a GitHub release tagged `vX.Y.Z`. Publishing the GitHub release
   triggers `.github/workflows/release.yml`.
6. Approve the protected `pypi` environment, then verify the PyPI page and a
   fresh installation:

   ```bash
   python -m venv /tmp/hamlet-release-check
   /tmp/hamlet-release-check/bin/pip install hamlet-toolkit==X.Y.Z
   /tmp/hamlet-release-check/bin/hamlet modes
   ```

PyPI does not permit replacing files for an existing version. If anything is
wrong, increment the patch version and make a new release.
