# Publishing HamLeT to PyPI

PyPI is the index `pip install` downloads from. Publishing there means anyone
can run `pip install hamlet-toolkit` instead of cloning this repository.

This guide assumes you have never published a package. Do the **one-time
setup** once; after that, every release is the **release checklist**.

A note on how this is wired: no password or API token is stored anywhere.
GitHub proves to PyPI that a release really came from this repository, using a
mechanism called *Trusted Publishing*. That is why the setup below involves
telling PyPI which repository to trust, rather than copying a secret.

---

## One-time setup

### 1. Make a PyPI account

Go to <https://pypi.org/account/register/> and register. PyPI requires
two-factor authentication, so it will ask you to set up an authenticator app
(Google Authenticator, 1Password, Authy — any of them). Keep the recovery
codes it gives you.

### 2. Tell PyPI to trust this repository

The project name `hamlet-toolkit` is not registered yet, so you create what
PyPI calls a *pending publisher* — a rule that says "if GitHub says a release
came from this repository, accept it, and create the project the first time".

1. Go to <https://pypi.org/manage/account/publishing/>.
2. Under **Add a new pending publisher**, choose **GitHub** and fill in:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `hamlet-toolkit` |
   | Owner | `GretaLupi` |
   | Repository name | `hamlet-toolkit` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Click **Add**.

All five fields must match exactly, including the environment name. If any of
them is wrong the upload is rejected with a permissions error, which is the
single most common failure here.

### 3. Create the `pypi` environment on GitHub

1. Go to <https://github.com/GretaLupi/hamlet-toolkit/settings/environments>.
2. **New environment**, name it `pypi`, click **Configure environment**.
3. Tick **Required reviewers** and add yourself, then **Save**.

That last step means a release pauses and waits for you to press a button
before anything is uploaded. It is the stop button between "I tagged a
release" and "the whole world can download it", and it is worth having.

---

## Release checklist

### 1. Check it locally

```bash
pytest
ruff check src tests
rm -rf dist
python -m build
python -m twine check dist/*
```

All four must pass. There is no CI workflow, so this is the gate.

`python -m build` makes the two files that get uploaded: a `.whl` (what pip
installs) and a `.tar.gz` (the source). `twine check` verifies that PyPI will
be able to render the README. Note that PyPI resolves relative links against
`pypi.org`, so images and links in `README.md` must be absolute URLs —
`twine check` passes them either way, so this is on you.

### 2. Set the version

The version appears in three files and they must agree:

- `pyproject.toml` — `version = "0.1.0"`
- `src/hamlet/__init__.py` — `__version__ = "0.1.0"`
- `CITATION.cff` — `version: 0.1.0`

There is a test that fails if `pyproject.toml` and `__init__.py` disagree.

**A version number can never be reused.** If you publish `0.1.0` and then find
a typo, PyPI will not let you replace it — you publish `0.1.1` instead. Even
deleting a release does not free the number.

### 3. Write the changelog

Move the entries under `## [Unreleased]` in `CHANGELOG.md` into a new dated
section, for example `## [0.1.0] - 2026-09-11`, and leave `## [Unreleased]`
empty above it.

### 4. Commit and push

```bash
git add -A
git commit -m "Release 0.1.0"
git push origin main
```

### 5. Create the GitHub release

1. Go to <https://github.com/GretaLupi/hamlet-toolkit/releases/new>.
2. **Choose a tag** → type `v0.1.0` → **Create new tag on publish**.
3. Title it `v0.1.0`, and paste the changelog section into the description.
4. Click **Publish release**.

Publishing the release is what starts the upload. Nothing before this point
sends anything to PyPI.

### 6. Approve the upload

1. Go to the **Actions** tab. A run called *Publish to PyPI* will be waiting.
2. It builds the files, then stops at the `pypi` environment for your
   approval. Click **Review deployments** → tick `pypi` → **Approve and
   deploy**.

### 7. Check it worked

Wait a minute, then look at <https://pypi.org/project/hamlet-toolkit/> — the
page should show your README, with the logo and the pipeline figure.

Then install it somewhere clean, which is the only test that proves a user can
get it:

```bash
python -m venv /tmp/hamlet-release-check
/tmp/hamlet-release-check/bin/pip install hamlet-toolkit==0.1.0
/tmp/hamlet-release-check/bin/hamlet modes
/tmp/hamlet-release-check/bin/hamlet where
```

---

## If something goes wrong

**"invalid-publisher" or a permissions error during upload.** One of the five
fields in the pending publisher does not match. The environment name `pypi` is
the one people usually miss.

**The run finished but nothing appeared on PyPI.** Check that you approved the
`pypi` environment — the job waits indefinitely otherwise.

**The README looks wrong on the PyPI page.** Images with relative paths do not
load there. Make them absolute `https://raw.githubusercontent.com/...` URLs,
then release a new patch version.

**You want to rehearse first.** TestPyPI (<https://test.pypi.org>) is a
throwaway copy of PyPI. It needs its own account and its own pending
publisher, pointing at the same repository. It is worth doing if you are
nervous, but the approval gate in step 6 already means nothing is uploaded
without you pressing a button.
