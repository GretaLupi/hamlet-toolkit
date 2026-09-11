"""What the device report says when there is no GPU, which is most machines.

The interesting cases are all failures to find a card, and they have different
causes that call for different actions -- a Windows wheel that cannot use one
at all, a CUDA build missing its runtime, a driver that is not loaded. The
report is the only place a user learns which of those they have, so these
tests pin the wording that distinguishes them.

TensorFlow is faked rather than installed in each configuration: the real one
reports whatever this machine happens to be, which is exactly one of the cases
under test and not the interesting one.
"""

from __future__ import annotations

import sys
import types

import pytest

from hamlet import compute


def _fake_tensorflow(*, gpus=(), built_with_cuda=True, cuda_raises=False):
    """A TensorFlow stand-in answering only what describe_compute() asks."""
    module = types.ModuleType("tensorflow")

    def is_built_with_cuda():
        if cuda_raises:
            raise RuntimeError("stripped build")
        return built_with_cuda

    module.test = types.SimpleNamespace(is_built_with_cuda=is_built_with_cuda)
    module.config = types.SimpleNamespace(
        list_physical_devices=lambda kind: list(gpus) if kind == "GPU" else [],
        experimental=types.SimpleNamespace(
            get_device_details=lambda device: {"device_name": device.name}
        ),
    )
    return module


@pytest.fixture
def fake_tf(monkeypatch):
    def install(**kwargs):
        monkeypatch.setitem(sys.modules, "tensorflow", _fake_tensorflow(**kwargs))

    return install


def _notes(report) -> str:
    return "\n".join(report.notes)


def test_windows_is_told_the_wheel_cannot_use_a_gpu_at_all(fake_tf, monkeypatch):
    """The report has to say this, or it reads as a driver problem.

    A Windows user with a working card, a current driver and a CPU-only wheel
    sees "no GPU visible" and reasonably concludes something is misconfigured.
    Nothing is: TensorFlow dropped native Windows GPU support at 2.11.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    fake_tf(built_with_cuda=False)

    notes = _notes(compute.describe_compute())

    assert "2.11" in notes
    assert "WSL2" in notes
    # And it must not send them driver-hunting instead.
    assert "expected rather than a fault" in notes


def test_a_cuda_build_with_no_card_blames_the_runtime_not_the_wheel(fake_tf, monkeypatch):
    """The Linux wheel is built with CUDA but ships no CUDA runtime.

    So "no GPU" here means the driver or the libraries are missing, which is
    fixable, and telling the user to reinstall TensorFlow would not fix it.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    fake_tf(built_with_cuda=True)

    notes = _notes(compute.describe_compute())

    assert "missing at runtime rather than unsupported" in notes
    assert "hamlet-toolkit[gpu]" in notes
    assert "driver has to come from the system" in notes


def test_a_non_cuda_build_on_linux_is_told_to_reinstall(fake_tf, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    fake_tf(built_with_cuda=False)

    notes = _notes(compute.describe_compute())

    assert "built without CUDA" in notes
    assert "hamlet-toolkit[gpu]" in notes


def test_macos_is_not_pointed_at_a_cuda_install_it_cannot_use(fake_tf, monkeypatch):
    """The gpu extra is Linux-only, so recommending it here would be a dead end."""
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_tf(built_with_cuda=False)

    notes = _notes(compute.describe_compute())

    assert "macOS has no CUDA path" in notes
    assert "hamlet-toolkit[gpu]" not in notes


def test_a_visible_gpu_gets_no_explanation_of_why_there_is_none(fake_tf, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    fake_tf(gpus=[types.SimpleNamespace(name="/physical_device:GPU:0")])

    report = compute.describe_compute()

    assert report.has_gpu
    notes = _notes(report)
    assert "sees no GPU" not in notes
    assert "WSL2" not in notes
    # The standing advice about which stage a GPU helps is still worth saying.
    assert "Dataset generation" in notes


def test_a_build_that_will_not_answer_gets_no_invented_reason(fake_tf, monkeypatch):
    """`is_built_with_cuda` is a nicety; an unknown cause is left unstated."""
    monkeypatch.setattr(sys, "platform", "linux")
    fake_tf(cuda_raises=True)

    notes = _notes(compute.describe_compute())

    assert "sees no GPU" in notes
    assert "built without CUDA" not in notes
    assert "missing at runtime" not in notes


def test_the_report_survives_tensorflow_not_being_installed(monkeypatch):
    """A base install has no TensorFlow and still needs a usable answer."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def refuse(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("no tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr("builtins.__import__", refuse)

    report = compute.describe_compute()

    assert not report.tensorflow_available
    assert not report.has_gpu
    assert report.cpu_count >= 1
    assert "not importable" in _notes(report)


# --- cluster access is key-based, by construction ---------------------------

def test_ssh_never_asks_for_anything():
    """A prompt on a worker thread has nobody to answer it.

    Every cluster operation runs in the background with no terminal attached,
    so an ssh that decides to ask for a password or a passphrase would wait
    forever and the job would look hung. BatchMode turns that into an error
    that names the fix.
    """
    from hamlet.cluster import DEFAULT_SSH_COMMAND, ClusterConfig, get_profile

    assert "BatchMode=yes" in DEFAULT_SSH_COMMAND
    assert any(item.startswith("ConnectTimeout") for item in DEFAULT_SSH_COMMAND)

    config = ClusterConfig(remote_dir="/x", scheduler=get_profile("slurm"), host="you@x")
    assert "BatchMode=yes" in config.ssh_command


def test_a_refused_key_is_told_apart_from_an_unreachable_host():
    """They need different fixes, and the generic message sends you to the wrong one."""
    from hamlet.cluster import _looks_like_an_auth_refusal

    assert _looks_like_an_auth_refusal("you@x: Permission denied (publickey,password).")
    assert _looks_like_an_auth_refusal(
        "Host key verification failed."
    )
    assert not _looks_like_an_auth_refusal(
        "ssh: connect to host x port 22: Connection timed out"
    )
    assert not _looks_like_an_auth_refusal("bash: sbatch: command not found")


def test_the_key_hint_gives_the_commands_to_run():
    from hamlet.cluster import _connection_hint

    hint = _connection_hint(False, True, "you@cluster.example.edu")
    assert "ssh-copy-id you@cluster.example.edu" in hint
    assert "ssh-keygen" in hint
    assert "ssh-agent" in hint

    unreachable = _connection_hint(False, False, "you@cluster.example.edu")
    assert "ssh-copy-id" not in unreachable, "wrong fix for an unreachable host"

    assert _connection_hint(True, False, "you@x") is None


def test_a_custom_ssh_command_is_still_honoured(tmp_path):
    """A site with a jump host or a wrapper keeps control of its own ssh."""
    from hamlet.cluster import ClusterConfig

    config = ClusterConfig.from_mapping({
        "remote_dir": "/x",
        "host": "you@x",
        "scheduler": "slurm",
        "ssh_command": "ssh -J bastion",
    })
    assert config.ssh_command == ("ssh", "-J", "bastion")


# --- the GPU choice exists only where it can be honoured ---------------------

def test_a_gpu_is_offered_on_linux_only(monkeypatch):
    """TensorFlow has no GPU build for Windows and no CUDA on macOS.

    Offering a choice that can never be honoured -- even labelled -- is an
    invitation to spend an afternoon on drivers.
    """
    from hamlet.gui import api

    monkeypatch.setattr(api.sys, "platform", "linux")
    assert "gpu" in {d["name"] for d in api.describe_compute_options()["devices"]}

    for platform in ("win32", "darwin"):
        monkeypatch.setattr(api.sys, "platform", platform)
        options = api.describe_compute_options()
        names = {d["name"] for d in options["devices"]}
        assert names == {"auto", "cpu"}, f"{platform} was offered {names}"
        assert options["gpu_possible_here"] is False
        # And says where a GPU is actually reachable from here.
        assert "cluster" in options["gpu_elsewhere_note"]
