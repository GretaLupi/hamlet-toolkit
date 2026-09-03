# Adding physical systems without weakening the workflow

The current package has a stable first vertical slice and a small
experiment-mode registry, but not yet a universal system-integration registry.
This distinction prevents a new physical system from being
represented with misleading Heisenberg-specific fields merely to reuse code.

## Reuse levels

### New local spin-chain Hamiltonian with site spectra

If the observation still has axes `(sample, site, bias)` and local targets can
be inferred from fixed neighboring-site windows, most of the package can be
reused: measurement import, DMRGPy unit conversion, preprocessing, grouped
splits, models, metrics, artifact mechanics, OOD profiles, and decision files.

Implement and test:

1. an immutable system family and parameter sampler;
2. backend Hamiltonian construction;
3. a supervised-view transformation with unambiguous target ownership;
4. metadata identifying observable, target names/units, interaction range, and
   boundary convention;
5. a reconstruction/analyzer rule for variable chain lengths;
6. positive and negative identifiability tests;
7. physical nuisance and held-out-chain validation;
8. artifact/advisor compatibility tests.

### Homogeneous or fixed-length global spin system

Global dataset views and supervised training already work. A new system still
needs a system-specific experimental analyzer, target-labelled report, fixed
site-count compatibility check, and identifiability study as parameters are
added. The current local experimental analyzer must not be reused.

### Different observation geometry, such as QPI maps

QPI should become the second vertical slice and drive the generalization. It
needs named spatial/momentum and energy axes, masks, arbitrary target names and
units, Fourier/preprocessing provenance, map-aware models, baselines, an
analyzer, and report sections. The current `SpectroscopyDataset.targets_mev`
and bond table are intentionally not presented as generic enough for this.

## Stable contracts worth preserving

- public physical units and explicit conversion at backend boundaries;
- raw portable datasets separated from cutoff-specific trained weights;
- one resolved preprocessing contract stored with every artifact;
- physical-sample grouped splits;
- untouched held-out test metrics;
- label-free experiment/training compatibility before prediction;
- machine-readable workflow decisions and human-readable reports;
- refusal to overwrite or mix incompatible run directories;
- explicit development (`quick`) versus distributable (`standard`/`research`)
  artifacts.

## Registry milestone

The first registry layer now maps a user-facing mode/variant to a stable system
identifier, supervised view, axes, primary channel, site-count rule, and
experimental-support status. This is sufficient for guided inspection without
claiming that training and reporting are fully generic.

After the second vertical slice establishes the real common interface, extend
the registry so each integration also provides
its dataset schema/adapter, supported supervised views, experiment
compatibility rules, analyzer, and report renderer. At that point
`HamiltonianLearningProject` can remain generic while physics-specific code
stays in registered integrations.

Building this registry before QPI would risk designing it around only the
Heisenberg assumptions. For now, adding new model-bank entries and related
spin-chain variants is safe; claiming arbitrary-system plug-ins is not.
