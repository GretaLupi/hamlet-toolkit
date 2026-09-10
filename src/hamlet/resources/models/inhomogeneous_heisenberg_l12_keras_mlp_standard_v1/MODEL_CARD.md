# Model card — `inhomogeneous_heisenberg_l12_keras_mlp_standard_v1`

Reference local-view model for the `inhomogeneous_heisenberg` system at
L = 12, recovering **bond-resolved** exchange couplings from spatially resolved
STM spectroscopy.

## Provenance and attribution

This is the trained model and dataset from the nanographene spin-chain study,
reused here rather than retrained, so that the published artifact is the one the
work was done with:

> *Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains*
> Greta Lupi, Saketh Ravuri, Chenxiao Zhao, Weidan Zhang, Cesare Roncaglia,
> Renxiang Liu, Xinliang Feng, Daniele Passerone, Pascal Ruffieux, Roman Fasel,
> Jose L. Lado, and Gonçalo Catarina (2026).

```bibtex
@article{lupi2026,
  title={Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene Spin Chains},
  author={Lupi, Greta and Ravuri, Saketh and Zhao, Chenxiao and Zhang, Weidan
          and Roncaglia, Cesare and Liu, Renxiang and Feng, Xinliang and
          Passerone, Daniele and Ruffieux, Pascal and Fasel, Roman and
          Lado, Jose L. and Catarina, Gon{\c{c}}alo},
  year={2026}
}
```

Source datasets: `dataset_30_40.npz` and `dataset_35_45.npz` from
[Inhomogeneous-Heisenberg-HL](https://github.com/GretaLupi/Inhomogeneous-Heisenberg-HL),
MIT licensed. **Please cite the paper above if you use this model.**

## What it does

Takes a **three-site sliding window** of a dI/dV map and returns the two
exchange couplings on the bonds inside that window, `J_left` and `J_right`, in
meV. Sliding the window along the chain reconstructs the full bond-resolved
coupling profile `J_1 … J_11`.

Because the input is a local window rather than the whole chain, this model is
**not restricted to a single chain length** the way a global-view model is: it
was trained on L = 12 chains but applies to any chain long enough to contain a
window. This is the practical difference between the local and global views.

## Training data

- 3000 simulated chains of 12 sites (11 bonds), yielding 30000 windows
- observable: integrated local susceptibility, i.e. dI/dV-like
- bias window: 0 to 50 meV over 200 points (source data runs to 100 meV)
- couplings sampled over **30 to 45 meV**
- experiment-like augmentation applied during training: additive noise 0.002,
  stochastic broadening of 0.5 to 2.0 points, offsets of 0.009 to 0.016, and a
  linear drift term

Sampled coupling range — **predictions outside it are extrapolation**:

| parameter | min [meV] | max [meV] |
|---|---|---|
| `J_left` | 30.0 | 45.0 |
| `J_right` | 30.0 | 45.0 |

## Accuracy

Keras MLP, preset `standard`, three seeds, ensembled by **median** (chosen on
validation MAE over mean and validation-weighted alternatives). Grouped split by
simulated chain, so no chain appears in more than one partition: 2100 train /
600 validation / 300 test chains.

Per bond on the 300 held-out chains (3000 windows), with skill against a
training-mean baseline (`1 - MAE_model / MAE_mean`):

| parameter | test MAE [meV] | baseline MAE [meV] | skill | correlation |
|---|---|---|---|---|
| `J_left` | 1.400 | 3.079 | 0.55 | 0.878 |
| `J_right` | 1.407 | 3.087 | 0.54 | 0.876 |

Overall ensemble MAE **1.404 meV** on the raw held-out data.

The manifest records 1.370 meV for the same split. That is not an
inconsistency: the manifest figure was computed on **augmented** spectra, the
distribution the model was trained and validated against, while the table above
is on the **raw** simulated data. Both are reported because they answer
different questions — how it does on data resembling a measurement, and how it
does on the clean simulation.

## When NOT to reuse this model

- the measurement does not cover 0 to 50 meV
- the couplings are expected outside 30 to 45 meV
- the experiment is not the `inhomogeneous_heisenberg` system, or not the
  `local_bonds` view
- the chain is shorter than one three-site window
- the observable is not the dI/dV-like integrated local susceptibility

`hamlet advise` checks the stored system, view, cutoff, and preprocessing
coverage against your measurement, and refuses a mismatch rather than
substituting a different cutoff.

## Requires TensorFlow

Unlike the other published models, which are scikit-learn estimators, this one
is a Keras ensemble. Loading it needs the optional extra:

```bash
pip install "hamlet-toolkit[ml]"
```

## Honest limits

- **Evaluated on one split.** The homogeneous reference model in this
  repository is benchmarked over five resampled splits, because per-parameter
  numbers at these dataset sizes move noticeably between splits. This artifact
  reports the single split it was trained with, which is what the published
  work used. Treat the per-bond figures as indicative rather than as tight
  bounds, and note the two bonds agreeing to within 0.007 meV is a reassuring
  internal consistency check rather than an independent replication.
- Accuracy is against **simulated** spectra from the generator that produced
  the training set. It quantifies the inverse problem, not whether the model
  Hamiltonian describes any particular material.
- Ensemble spread at inference measures agreement between the three seeds, not
  distance from truth. It is not a calibrated uncertainty.
- A skill of ~0.55 means the model roughly halves the error against guessing
  the training mean. That is a useful signal, not a precision measurement of
  individual bonds; the paper's reconstructions rely on the whole profile
  rather than any single bond.

## Reproducing

The evaluation table above is reproducible from the source datasets and this
artifact using the split seed recorded in `manifest.json`
(`training_preset.split_seed = 42`); the per-bond numbers are also stored in
`held_out_evaluation.json`.
