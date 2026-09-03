"""Checkpointed and cache-safe generation of portable spectroscopy datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import numpy as np

from ..branding import brand_manifest
from .dataset import SpectroscopyDataset
from .generation import SystemFamily, generate_dataset
from ..simulation import SpectroscopyProtocol, SpectroscopySimulator


@dataclass(frozen=True)
class CheckpointedGenerationResult:
    dataset: SpectroscopyDataset
    dataset_path: Path
    manifest_path: Path
    cache_hit: bool
    resumed_chunks: int
    generated_chunks: int


def generate_dataset_checkpointed(
    family: SystemFamily,
    simulator: SpectroscopySimulator,
    protocol: SpectroscopyProtocol,
    *,
    n_samples: int,
    output_path: str | Path,
    recipe: Mapping[str, Any],
    seed: int = 42,
    checkpoint_every: int = 25,
    progress: Callable[[int, int], None] | None = None,
) -> CheckpointedGenerationResult:
    """Generate, resume, and cache a dataset under an exact recipe fingerprint.

    Completed chunks are portable NPZ files. They are removed only after the
    final dataset has been written atomically. A matching completed dataset is
    loaded without invoking the simulator; a mismatched recipe is rejected.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    destination = Path(output_path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("generated dataset output_path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.with_suffix(".generation.json")
    checkpoint_dir = destination.parent / f".{destination.stem}.checkpoints"
    resolved_recipe = _jsonable(dict(recipe))
    fingerprint = _fingerprint(resolved_recipe)
    manifest = {
        "toolkit": brand_manifest(),
        "generation_schema_version": 1,
        "fingerprint": fingerprint,
        "recipe": resolved_recipe,
    }

    if destination.exists():
        _require_matching_manifest(manifest_path, fingerprint)
        dataset = SpectroscopyDataset.load(destination)
        if dataset.n_samples != n_samples:
            raise ValueError("cached dataset sample count does not match its recipe")
        if progress is not None:
            progress(n_samples, n_samples)
        return CheckpointedGenerationResult(
            dataset, destination, manifest_path, True, 0, 0
        )

    if manifest_path.exists():
        _require_matching_manifest(manifest_path, fingerprint)
    else:
        _write_json_atomic(manifest_path, manifest)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sizes = [
        min(checkpoint_every, n_samples - start)
        for start in range(0, n_samples, checkpoint_every)
    ]
    child_sequences = np.random.SeedSequence(seed).spawn(len(sizes))
    chunk_seeds = [int(item.generate_state(1, dtype=np.uint32)[0]) for item in child_sequences]
    chunks: list[SpectroscopyDataset] = []
    resumed_chunks = 0
    generated_chunks = 0
    completed = 0

    for index, (size, chunk_seed) in enumerate(zip(sizes, chunk_seeds)):
        chunk_path = checkpoint_dir / f"chunk-{index:05d}.npz"
        if chunk_path.exists():
            chunk = SpectroscopyDataset.load(chunk_path)
            if chunk.n_samples != size:
                raise ValueError(f"checkpoint has wrong sample count: {chunk_path}")
            resumed_chunks += 1
            completed += size
            if progress is not None:
                progress(completed, n_samples)
        else:
            base = completed
            chunk = generate_dataset(
                family,
                simulator,
                protocol,
                size,
                seed=chunk_seed,
                progress=(
                    (lambda done, _total, offset=base: progress(offset + done, n_samples))
                    if progress is not None
                    else None
                ),
            )
            temporary = chunk_path.with_suffix(".partial.npz")
            chunk.save(temporary)
            os.replace(temporary, chunk_path)
            generated_chunks += 1
            completed += size
        chunks.append(chunk)

    dataset = _combine_chunks(
        chunks,
        recipe=resolved_recipe,
        fingerprint=fingerprint,
        chunk_seeds=chunk_seeds,
    )
    temporary_dataset = destination.with_suffix(".partial.npz")
    dataset.save(temporary_dataset)
    os.replace(temporary_dataset, destination)
    shutil.rmtree(checkpoint_dir)
    return CheckpointedGenerationResult(
        dataset,
        destination,
        manifest_path,
        False,
        resumed_chunks,
        generated_chunks,
    )


def _combine_chunks(
    chunks: list[SpectroscopyDataset],
    *,
    recipe: Mapping[str, Any],
    fingerprint: str,
    chunk_seeds: list[int],
) -> SpectroscopyDataset:
    if not chunks:
        raise ValueError("no generated chunks")
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk.system_type != first.system_type:
            raise ValueError("generated chunks contain different systems")
        if chunk.target_names != first.target_names:
            raise ValueError("generated chunks contain different targets")
        if not np.array_equal(chunk.bias_mev, first.bias_mev):
            raise ValueError("generated chunks contain different bias grids")
    metadata = dict(first.metadata)
    metadata.update(
        {
            "generation_recipe": dict(recipe),
            "generation_fingerprint": fingerprint,
            "chunk_seeds": chunk_seeds,
            "n_samples": int(sum(chunk.n_samples for chunk in chunks)),
        }
    )
    return SpectroscopyDataset(
        spectra=np.concatenate([chunk.spectra for chunk in chunks], axis=0),
        targets_mev=np.concatenate([chunk.targets_mev for chunk in chunks], axis=0),
        bias_mev=first.bias_mev,
        target_names=first.target_names,
        system_type=first.system_type,
        metadata=metadata,
    )


def _require_matching_manifest(path: Path, fingerprint: str) -> None:
    if not path.exists():
        raise FileExistsError(
            f"generated dataset exists without a recipe manifest: {path}; "
            "choose a new output path"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise FileExistsError(
            f"generation recipe does not match existing state at {path}; "
            "choose a new output path"
        )


def _fingerprint(recipe: Mapping[str, Any]) -> str:
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
