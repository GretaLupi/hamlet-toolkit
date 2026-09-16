"""Where the heavy work runs: CPU, GPU, or somebody else's cluster.

Two things about this package's cost profile decide everything here, and both
are easy to get wrong:

* **Dataset generation is not GPU work.** It is DMRG and exact diagonalisation
  through DMRGPy, which are CPU-bound and single-threaded per chain. A GPU does
  nothing for it. Generation is also the expensive stage -- hours, against
  minutes for training -- so "put it on the GPU" is the wrong instinct for the
  part that actually hurts. Cores and parallel chains are what help.
* **Training is GPU work, but only for the neural models.** ``keras_mlp`` and
  ``keras_cnn`` use one if it is visible. ``ridge`` and ``random_forest`` are
  scikit-learn and will not, however many GPUs are present.

So this module reports what is available and lets a device be chosen, and is
careful to say which stage each choice affects. Claiming otherwise would send
people to buy GPU time for a stage that cannot use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Models that can actually use an accelerator. Kept here rather than inferred,
# because "is it a neural model" is a fact about this package's registry, not
# something to guess from a name.
GPU_CAPABLE_MODELS = frozenset({"keras_mlp", "keras_cnn"})


# --- importing TensorFlow ---------------------------------------------------
#
# TensorFlow's own `and-cuda` wheels ship libcusolver in
# `nvidia/cusolver/lib`, and that directory is in the RUNPATH of
# libtensorflow_cc.so.2 but not of libtensorflow_framework.so.2, which is where
# the load actually happens. The loader therefore never finds it, TensorFlow
# logs "Cannot dlopen some GPU libraries" and reports no GPU at all -- on a
# machine whose driver and CUDA wheels are both fine. Every other CUDA library
# resolves; removing them one at a time shows cusolver is the only one that
# matters (reported by @joselado, issue #1).
#
# Loading it into the process before TensorFlow is imported fixes it with no
# environment variable and no change for the user. Every import of TensorFlow
# or Keras in this package goes through the two helpers below so that this
# cannot be forgotten at a new call site.


def preload_cuda_libraries() -> tuple[str, ...]:
    """Put TensorFlow's own CUDA libraries where its loader will find them.

    Returns the libraries loaded, for diagnostics. Does nothing at all when the
    pip CUDA wheels are absent, when TensorFlow has already been imported (too
    late to matter), or off Linux, so it is safe to call unconditionally.
    """
    import ctypes
    import importlib.util
    import sys
    from pathlib import Path as _Path

    if sys.platform != "linux" or "tensorflow" in sys.modules:
        return ()
    try:
        spec = importlib.util.find_spec("nvidia.cusolver")
    except (ImportError, ValueError):  # pragma: no cover - absent is the norm
        return ()
    if spec is None or not spec.submodule_search_locations:
        return ()
    loaded: list[str] = []
    directory = _Path(next(iter(spec.submodule_search_locations))) / "lib"
    for library in sorted(directory.glob("libcusolver.so.*")):
        try:
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        except OSError:  # pragma: no cover - a broken wheel is not our failure
            continue
        loaded.append(str(library))
    return tuple(loaded)


def load_tensorflow():
    """Import TensorFlow with its CUDA libraries reachable."""
    preload_cuda_libraries()
    import tensorflow as tf

    return tf


def load_keras():
    """Import Keras with TensorFlow's CUDA libraries reachable."""
    preload_cuda_libraries()
    import keras

    return keras


def _why_no_gpu(built_with_cuda: bool | None) -> list[str]:
    """Why a working TensorFlow sees no GPU, which is not the same on every OS.

    The bare fact -- "no GPU visible" -- reads as a driver or hardware fault
    everywhere, and on native Windows it is neither: the wheel there is
    CPU-only and has been since TensorFlow 2.11, so no driver, CUDA install or
    environment variable will ever expose the card. Reporting the cause is the
    difference between a one-line answer and an afternoon spent updating
    drivers that were never the problem.
    """
    import sys

    if sys.platform == "win32":
        return [
            "On native Windows that is expected rather than a fault: "
            "TensorFlow has shipped no Windows GPU support since 2.11, and its "
            "Windows wheel is CPU-only whatever the driver reports. Training a "
            "Keras model on the card needs WSL2, with "
            '`pip install "hamlet-toolkit[gpu]"` inside it.'
        ]
    if sys.platform == "darwin":
        return [
            "macOS has no CUDA path at all. Apple's tensorflow-metal plugin is "
            "the only accelerator option there, and HamLeT neither requires "
            "nor tests it."
        ]
    if built_with_cuda is False:
        return [
            "This TensorFlow was built without CUDA, so no driver or "
            "environment change will expose a GPU to it. "
            '`pip install "hamlet-toolkit[gpu]"` installs one that was.'
        ]
    if built_with_cuda:
        return [
            "This TensorFlow is built with CUDA, so the card is missing at "
            "runtime rather than unsupported. HamLeT already preloads the "
            "CUDA libraries that ship in the pip wheels, so the likely cause "
            "is the part pip cannot provide: no NVIDIA driver is loaded. "
            "`nvidia-smi` says whether one is."
        ]
    return []


@dataclass(frozen=True)
class Accelerator:
    """One visible compute device."""

    kind: str
    name: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "detail": self.detail}


@dataclass(frozen=True)
class ComputeReport:
    """What this machine can run, and what each part of the workflow will use."""

    cpu_count: int
    accelerators: tuple[Accelerator, ...] = ()
    tensorflow_available: bool = False
    notes: tuple[str, ...] = ()

    @property
    def has_gpu(self) -> bool:
        return any(item.kind == "gpu" for item in self.accelerators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "has_gpu": self.has_gpu,
            "accelerators": [item.to_dict() for item in self.accelerators],
            "tensorflow_available": self.tensorflow_available,
            "gpu_capable_models": sorted(GPU_CAPABLE_MODELS),
            "notes": list(self.notes),
        }


def describe_compute() -> ComputeReport:
    """Report the devices available, without committing to using any of them.

    Importing TensorFlow is the only way to ask it what it can see, and that
    import is slow and noisy, so failure to import is reported rather than
    raised: a machine with no ML extra still has a usable answer.
    """
    import os

    cpu_count = os.cpu_count() or 1
    accelerators: list[Accelerator] = []
    notes: list[str] = []
    tensorflow_available = False
    built_with_cuda: bool | None = None

    try:
        tf = load_tensorflow()

        tensorflow_available = True
        try:
            built_with_cuda = bool(tf.test.is_built_with_cuda())
        except Exception:  # noqa: BLE001 - a nicety on stripped or older builds
            built_with_cuda = None
        for device in tf.config.list_physical_devices("GPU"):
            detail = ""
            try:
                info = tf.config.experimental.get_device_details(device)
                name = info.get("device_name") or device.name
                capability = info.get("compute_capability")
                detail = f"compute capability {capability}" if capability else ""
            except Exception:  # noqa: BLE001 - details are a nicety, not a contract
                name = device.name
            accelerators.append(Accelerator("gpu", str(name), detail))
    except Exception:  # noqa: BLE001 - absent or broken, both mean "no GPU here"
        notes.append(
            "TensorFlow is not importable, so no GPU can be used and the Keras "
            "models are unavailable. Install the ml extra to change that."
        )

    if tensorflow_available and not accelerators:
        notes.append(
            "TensorFlow is installed but sees no GPU. Training will use the CPU."
        )
        notes.extend(_why_no_gpu(built_with_cuda))
    notes.append(
        "Dataset generation is DMRG and exact diagonalisation on the CPU; a GPU "
        "does not help it. Generation is also the long stage, so more cores, "
        "not a faster accelerator, is what shortens it."
    )
    notes.append(
        f"Only {', '.join(sorted(GPU_CAPABLE_MODELS))} can use a GPU; ridge and "
        "random_forest are scikit-learn and run on the CPU either way."
    )
    return ComputeReport(
        cpu_count=cpu_count,
        accelerators=tuple(accelerators),
        tensorflow_available=tensorflow_available,
        notes=tuple(notes),
    )


def verify_accelerator(report: "ComputeReport | None" = None) -> dict[str, Any]:
    """Actually train one convolution step on the GPU, rather than trusting the list.

    A visible device proves the driver is loaded and nothing else. Two failures
    survive that check and only appear once real work starts, both reported by
    @joselado on a GTX 1060 (issue #1): XLA's autotuner finding no supported
    configuration for a convolution, and a cuDNN too new to still support the
    card. Either one kills a training run after the dataset has been generated,
    which is the most expensive moment to discover it.

    A Conv1D is the right probe because convolution is what breaks; a dense
    layer passes on cards where ``keras_cnn`` cannot run. Returns what
    happened, with advice when it fails, and never raises.
    """
    report = report if report is not None else describe_compute()
    if not report.accelerators:
        return {"ran": False, "ok": None, "reason": "no GPU is visible to TensorFlow"}
    try:
        import numpy as np

        keras = load_keras()
        tf = load_tensorflow()
        with tf.device("/GPU:0"):
            probe = keras.Sequential(
                [
                    keras.Input((16, 1)),
                    keras.layers.Conv1D(4, 3, padding="same", activation="relu"),
                    keras.layers.Flatten(),
                    keras.layers.Dense(1),
                ]
            )
            probe.compile(optimizer="adam", loss="mse", jit_compile=False)
            probe.fit(
                np.zeros((8, 16, 1), dtype="float32"),
                np.zeros((8, 1), dtype="float32"),
                epochs=1,
                batch_size=8,
                verbose=0,
            )
    except BaseException as exc:  # noqa: BLE001 - a broken GPU stack raises anything
        message = f"{type(exc).__name__}: {exc}"
        return {
            "ran": True,
            "ok": False,
            "error": message,
            "advice": _accelerator_advice(message),
        }
    return {"ran": True, "ok": True, "error": None, "advice": ()}


def _accelerator_advice(message: str) -> tuple[str, ...]:
    """What to try, for the failures that have actually been seen."""
    lowered = message.lower()
    advice: list[str] = []
    if "autotun" in lowered or "xla" in lowered:
        advice.append(
            "XLA could not compile a convolution for this card. HamLeT keeps "
            "XLA off by default, so something has turned it back on: check for "
            "jit_compile in your model options."
        )
    if "cudnn" in lowered or "5003" in lowered or "convolution" in lowered:
        advice.append(
            "cuDNN refused a convolution on this card. Recent cuDNN releases "
            "have dropped support for older GPUs; the version TensorFlow 2.21 "
            "is built against still works: "
            'pip install "nvidia-cudnn-cu12==9.3.0.75"'
        )
    if "out of memory" in lowered or "oom" in lowered:
        advice.append(
            "The card ran out of memory on a tiny probe, so something else is "
            "using it. Check nvidia-smi."
        )
    if not advice:
        advice.append(
            "The GPU is visible but cannot run a convolution. Training with "
            "device: cpu will work; the Keras models are the only ones that "
            "would have used the card."
        )
    return tuple(advice)


def gpu_unavailable_summary(report: "ComputeReport") -> str:
    """One line for a card or a label, where the full note will not fit.

    Empty when a GPU is there. The wording distinguishes "cannot on this
    platform" from "not present here", because only the second is something a
    user can go and fix.
    """
    import sys

    if report.has_gpu:
        return ""
    if not report.tensorflow_available:
        return "TensorFlow is not installed, so no GPU can be used."
    if sys.platform == "win32":
        return (
            "Not possible on native Windows: TensorFlow has shipped no Windows "
            "GPU build since 2.11. A card here needs WSL2."
        )
    if sys.platform == "darwin":
        return "Not possible on macOS: TensorFlow has no CUDA support there."
    return "No GPU is visible to TensorFlow on this machine."


@dataclass(frozen=True)
class DeviceRequest:
    """A choice of where training should run.

    ``"auto"`` uses a GPU when one is visible, which is TensorFlow's own
    behaviour and the right default. ``"cpu"`` is worth having explicitly: a
    small model on a busy shared GPU is often slower than the CPU, and someone
    else may need the card.
    """

    device: str = "auto"
    gpu_index: int | None = None
    memory_growth: bool = True

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError(
                f"device must be auto, cpu or gpu; got {self.device!r}"
            )
        if self.gpu_index is not None and self.gpu_index < 0:
            raise ValueError("gpu_index must not be negative")
        if self.gpu_index is not None and self.device == "cpu":
            raise ValueError("a gpu_index cannot be combined with device: cpu")


def configure_device(request: DeviceRequest | None = None) -> dict[str, Any]:
    """Apply a device choice to TensorFlow, and report what was actually done.

    Returns rather than raises when the request cannot be met, because a
    missing GPU is a reason to train on the CPU, not a reason to abandon a run
    that has already spent hours generating its dataset. The report says what
    happened so the artifact's provenance is not a guess.

    Must run before any model is built: TensorFlow fixes visible devices at
    initialisation, and both this and ``memory_growth`` are refused afterwards.
    """
    settings = request or DeviceRequest()
    outcome: dict[str, Any] = {
        "requested": settings.device,
        "gpu_index": settings.gpu_index,
        "used": "cpu",
        "changed": False,
        "warnings": [],
    }
    try:
        tf = load_tensorflow()
    except Exception:  # noqa: BLE001
        if settings.device == "gpu":
            outcome["warnings"].append(
                "a GPU was requested but TensorFlow is not importable; the CPU "
                "will be used"
            )
        return outcome

    gpus = tf.config.list_physical_devices("GPU")
    if settings.device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
            outcome["changed"] = True
        except RuntimeError as exc:
            # Already initialised. Reported rather than swallowed: the run is
            # about to use a device the caller asked it not to.
            outcome["warnings"].append(
                f"could not hide the GPU because TensorFlow is already "
                f"initialised: {exc}"
            )
        return outcome

    if not gpus:
        if settings.device == "gpu":
            outcome["warnings"].append(
                "a GPU was requested but none is visible; the CPU will be used"
            )
        return outcome

    chosen = gpus
    if settings.gpu_index is not None:
        if settings.gpu_index >= len(gpus):
            outcome["warnings"].append(
                f"GPU {settings.gpu_index} was requested but only "
                f"{len(gpus)} is/are visible; using the first"
            )
        else:
            chosen = [gpus[settings.gpu_index]]

    try:
        if settings.memory_growth:
            # Without this TensorFlow claims the whole card at startup, which
            # on a shared machine takes it away from everyone else for the
            # length of the run.
            for device in chosen:
                tf.config.experimental.set_memory_growth(device, True)
        if chosen is not gpus:
            tf.config.set_visible_devices(chosen, "GPU")
        outcome["changed"] = True
    except RuntimeError as exc:
        outcome["warnings"].append(
            f"device settings could not be applied because TensorFlow is "
            f"already initialised: {exc}"
        )

    outcome["used"] = "gpu"
    outcome["devices"] = [device.name for device in chosen]
    return outcome


def advise_device(model: str, report: ComputeReport | None = None) -> dict[str, Any]:
    """What a given model will actually use, and whether a GPU would help it."""
    current = report or describe_compute()
    capable = model in GPU_CAPABLE_MODELS
    if not capable:
        return {
            "model": model,
            "can_use_gpu": False,
            "will_use": "cpu",
            "why": (
                f"{model} is a scikit-learn model; it runs on the CPU whether or "
                "not a GPU is present"
            ),
        }
    if not current.tensorflow_available:
        return {
            "model": model,
            "can_use_gpu": True,
            "will_use": "unavailable",
            "why": f"{model} needs TensorFlow, which is not installed",
        }
    if not current.has_gpu:
        return {
            "model": model,
            "can_use_gpu": True,
            "will_use": "cpu",
            "why": "no GPU is visible to TensorFlow on this machine",
        }
    return {
        "model": model,
        "can_use_gpu": True,
        "will_use": "gpu",
        "why": (
            f"{model} will use "
            f"{current.accelerators[0].name} unless you choose otherwise"
        ),
    }


__all__ = [
    "Accelerator",
    "ComputeReport",
    "DeviceRequest",
    "GPU_CAPABLE_MODELS",
    "advise_device",
    "configure_device",
    "describe_compute",
    "gpu_unavailable_summary",
]
