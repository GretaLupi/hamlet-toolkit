"""Checkpointed and cache-safe generation of portable spectroscopy datasets.

Generation is the expensive stage -- hours, against minutes for training --
and it is embarrassingly parallel: every chain is an independent simulation.
It is parallelised here at the chunk level rather than the chain level because
chunks are already the unit of checkpointing, and because their seeds are
derived from the seed sequence by index, so a chunk's contents do not depend
on when or where it ran. That is what makes ``workers`` a pure execution
detail: the dataset is bit-identical whether one core produced it or twelve,
which is why it is excluded from the recipe fingerprint.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _isolate_worker(scratch_root: str) -> None:
    """Give each worker process its own directory to scribble in.

    DMRGPy derives its scratch paths from the working directory --
    ``os.getcwd() + "/.mpsfolder/"`` -- and chdirs into them while it solves.
    Two processes sharing a working directory therefore share those files and
    overwrite each other's wavefunctions, and they do it silently: the run
    completes and the dataset is quietly wrong, which is far worse than a
    crash. Since the working directory is process-global state, this also rules
    threads out entirely -- only separate processes can be isolated this way.
    """
    private = Path(scratch_root) / f"worker-{os.getpid()}"
    private.mkdir(parents=True, exist_ok=True)
    os.chdir(private)


def _generate_chunk_to_checkpoint(task: tuple[Any, ...]) -> tuple[int, int]:
    """Simulate one chunk in a worker and write its checkpoint.

    The chunk is saved here rather than returned so that the arrays never
    cross the process boundary, and so a worker's output survives a failure
    anywhere else in the run exactly as the serial path's does. Top-level
    because a worker has to be able to unpickle it under every start method,
    including the spawn used on Windows and macOS.
    """
    index, family, simulator, protocol, size, chunk_seed, chunk_path = task
    chunk = generate_dataset(family, simulator, protocol, size, seed=chunk_seed)
    _save_chunk_atomic(chunk, Path(chunk_path))
    return int(index), int(size)


def _save_chunk_atomic(chunk: SpectroscopyDataset, chunk_path: Path) -> None:
    temporary = chunk_path.with_suffix(".partial.npz")
    chunk.save(temporary)
    os.replace(temporary, chunk_path)


@dataclass(frozen=True)
class CheckpointedGenerationResult:
    dataset: SpectroscopyDataset
    dataset_path: Path
    manifest_path: Path
    cache_hit: bool
    resumed_chunks: int
    generated_chunks: int
    workers_used: int = 1


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
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> CheckpointedGenerationResult:
    """Generate, resume, and cache a dataset under an exact recipe fingerprint.

    Completed chunks are portable NPZ files. They are removed only after the
    final dataset has been written atomically. A matching completed dataset is
    loaded without invoking the simulator; a mismatched recipe is rejected.

    ``workers`` above one simulates that many chunks at once, each in its own
    process and its own working directory. The result does not depend on it.
    Progress then arrives per chunk instead of per chain, which is also the
    granularity at which such a run can be stopped.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
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
            dataset, destination, manifest_path, True, 0, 0, 1
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
    # Indexed rather than appended: chunks may now finish out of order, and the
    # dataset's sample order has to stay the recipe's, not the schedule's.
    chunks: list[SpectroscopyDataset | None] = [None] * len(sizes)
    resumed_chunks = 0
    generated_chunks = 0
    completed = 0
    pending: list[tuple[int, int, int]] = []

    # Everything already on disk is claimed first, so a resumed run credits
    # its existing work immediately instead of after the first new chunk.
    for index, (size, chunk_seed) in enumerate(zip(sizes, chunk_seeds)):
        chunk_path = checkpoint_dir / f"chunk-{index:05d}.npz"
        if chunk_path.exists():
            chunk = SpectroscopyDataset.load(chunk_path)
            if chunk.n_samples != size:
                raise ValueError(f"checkpoint has wrong sample count: {chunk_path}")
            chunks[index] = chunk
            resumed_chunks += 1
            completed += size
            if progress is not None:
                progress(completed, n_samples)
        else:
            pending.append((index, size, chunk_seed))

    # No more workers than there is work for them, or cores to run them on.
    # Asking for twelve on a four-core laptop is a request to make it slower.
    workers_used = max(1, min(int(workers), len(pending), os.cpu_count() or 1)) if pending else 1

    if workers_used > 1:
        scratch_root = checkpoint_dir / "scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)
        # Absolute, because a worker changes its working directory before it
        # simulates anything: a relative checkpoint path resolved there would
        # point inside the scratch directory, or nowhere at all.
        absolute_scratch = scratch_root.resolve()
        tasks = [
            (
                index,
                family,
                simulator,
                protocol,
                size,
                chunk_seed,
                str((checkpoint_dir / f"chunk-{index:05d}.npz").resolve()),
            )
            for index, size, chunk_seed in pending
        ]
        # The platform's default start method, deliberately, and not spawn.
        # Spawn would avoid a real hazard -- generation is started from the
        # browser interface's threaded server, and forking a threaded process
        # gives the child only the forking thread, so a lock another thread
        # held arrives locked -- but it re-imports `__main__` in every worker,
        # and both spawn and forkserver therefore fail outright wherever
        # `__main__` is not importable: a notebook, or `python -c`. This
        # package ships notebooks as its documented Python example, so that
        # trades a low-probability deadlock for a certain breakage of a
        # supported workflow. Note that on Windows and macOS the default *is*
        # spawn, which is why a script that generates a dataset needs the
        # usual `if __name__ == "__main__":` guard to be portable.
        pool = ProcessPoolExecutor(
            max_workers=workers_used,
            initializer=_isolate_worker,
            initargs=(str(absolute_scratch),),
        )
        try:
            futures = [pool.submit(_generate_chunk_to_checkpoint, task) for task in tasks]
            for future in as_completed(futures):
                index, size = future.result()
                chunks[index] = SpectroscopyDataset.load(
                    checkpoint_dir / f"chunk-{index:05d}.npz"
                )
                generated_chunks += 1
                completed += size
                if progress is not None:
                    # Raises to stop the run: chunk completion is therefore
                    # both the progress and the cancellation granularity.
                    progress(completed, n_samples)
        except BaseException:
            # A stop request, or a worker that failed. Queued chunks are
            # dropped rather than simulated to the end before the request is
            # honoured; the ones already running are allowed to finish, since
            # a process cannot be interrupted safely part-way through a write.
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    else:
        for index, size, chunk_seed in pending:
            chunk_path = checkpoint_dir / f"chunk-{index:05d}.npz"
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
            _save_chunk_atomic(chunk, chunk_path)
            chunks[index] = chunk
            generated_chunks += 1
            completed += size

    ordered = [chunk for chunk in chunks if chunk is not None]
    if len(ordered) != len(sizes):
        raise RuntimeError("a chunk was neither resumed nor generated")

    dataset = _combine_chunks(
        ordered,
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
        workers_used,
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
