# Architecture and migration plan

## First invariant

The initial estimator is local: three adjacent site spectra map to two adjacent
exchange couplings. The total chain length is therefore an inference-time
dimension, not a neural-network input dimension. For a chain with `L` sites:

- preprocessing produces an `(L, resolution)` spectral map;
- windowing produces `(L - 2, 3 * resolution)` model inputs;
- the model produces `(L - 2, 2)` local coupling estimates;
- overlap averaging produces `L - 1` bond couplings.

## Package boundaries

- `systems`: immutable physical specifications, with no ML or backend imports.
- `preprocessing`: the single shared training/inference transformation path.
- `inference`: model adapters and reconstruction of physical parameters.
- `measurements`: laboratory-independent named axes, channels, units, masks,
  uncertainty, metadata, and portable serialization.
- `experiments`: mode/variant registry, guided raw inspection, versioned
  experiment-project manifests, and canonical input resolution.
- `io`: recipe-driven instrument import plus compatibility CSV loaders.
- `benchmarks`: explicitly named statistical metrics.
- `simulation`: a backend contract and optional DMRGPy implementation.
- `data`: reproducible generation, portable metadata, and supervised views.
- `models`: lazy-loaded Keras and scikit-learn supervised regressors.
- `experimental`: high-level ensemble inference, quality checks, plots, and exports.
- `training`: grouped splits, augmentation, scaling, aggregation, and artifact manifests.
- `project`: configuration-driven orchestration and safe workflow state.
- `workflow`: versioned experiment/resource preflight decisions for CLI and a
  future UI.

## Migration from the reference repository

All public energies are expressed in meV. At the simulation boundary, the
DMRGPy adapter applies the project convention `1 DMRGPy energy unit = 10 meV`
to couplings, bias grids, and broadening. Returned dataset axes remain in meV.

| Reference code | Package destination |
|---|---|
| `generate_dataset.py` Hamiltonian construction | `systems` + future `simulation.dmrgpy` |
| `train_ensemble.py` preprocessing | `preprocessing` |
| `train_ensemble.py` window creation | `preprocessing.windows` |
| `predict_experimental_chains.py` CSV loading | `io.experimental` |
| `predict_experimental_chains.py` reconstruction | `inference` |
| `fidelity_noise_analysis.py` metrics | `benchmarks` |

The source-only checkout under `reference/` remains unchanged and is not part of
the installable distribution.

## Next milestones

1. Validate the frozen Heisenberg workflow across additional experimental chains.
2. Validate homogeneous global inference on a genuine generated DMRGPy dataset
   and representative fixed-length experimental chains.
3. Add a second vertical slice, beginning with the QPI/Fermi diffusion project.
4. Generalize project configuration through a system registry informed by both
   vertical slices; see `adding-physical-systems.md`.
5. Add exact diagonalization as a lightweight backend for small chains.
