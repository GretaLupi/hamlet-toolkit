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

    try:
        import tensorflow as tf

        tensorflow_available = True
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
        import tensorflow as tf
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
]
