# L=8 demo datasets

Small simulated datasets shipped with the repository so that
[`notebooks/04_l8_three_mode_workflow.ipynb`](../../notebooks/04_l8_three_mode_workflow.ipynb)
runs immediately after cloning, without a multi-hour simulation job.

| File | System | Chains | Targets | Observable |
|---|---|---:|---|---|
| `inhomogeneous_l8_ed.npz` | `inhomogeneous_heisenberg` | 64 | one `J` per bond (7) | `Sz` |
| `homogeneous_xxz_l8_ed.npz` | `homogeneous_xxz_j1j2j3` | 32 | `J1_xy, J2, J3, Jz` | `total_spin` |
| `homogeneous_xxz_dmi_l8_ed.npz` | `homogeneous_xxz_j1j2j3_dmi` | 32 | `J1_xy, J2, J3, Jz, D_z_magnitude` | `total_spin` |

All three are L=8 chains on a shared 0–60 meV grid with 61 bias points and 1.0 meV
broadening, produced by the DMRGPy backend with `dynamics_mode="ED"` (exact
diagonalization of the dynamical correlators, which is exact at this chain length).
Energies are meV throughout.

Each `.npz` has a `.generation.json` sidecar recording the complete resolved recipe and
its SHA-256 fingerprint. The package refuses to overwrite a completed dataset whose
recipe differs, so these files can be regenerated or extended reproducibly — see the
last section of the notebook for the exact generator call.

## These are pilots, not benchmarks

32–64 chains is small on purpose: at L=8, ED costs roughly 45 s per chain for a single
`Sz` correlator and 140 s per chain for `total_spin`, which evaluates `Sxx`, `Syy`, and
`Szz`. They are sized for a fast, reproducible demonstration of the workflow.

Use them to see how the pipeline behaves and how to read its diagnostics — not as
evidence of achievable accuracy. In particular the global (homogeneous) views get only
one training example per chain for four or five targets, so at this scale several
parameters carry no skill over a training-mean baseline. The notebook quantifies this
per parameter rather than hiding it.
