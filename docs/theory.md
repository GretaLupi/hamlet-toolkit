# The physics behind HamLeT

What the package computes, what it assumes, and what the numbers it returns do
and do not mean. Every formula here is the one implemented in
[`src/hamlet/simulation/dmrgpy.py`](../src/hamlet/simulation/dmrgpy.py); the
Hamiltonian blocks in the PDF reports come from the same source.

Energies are in meV throughout. At the DMRGPy boundary the package converts:
**1 DMRGPy energy unit = 10 meV**.

## 1. The inverse problem

A spin chain's excitation spectrum is a function of its exchange couplings.
Computing the spectrum from the couplings is the forward problem, and it is
well posed: given **J**, simulation returns the spectrum *S*. Experiments pose
the reverse question — given a measured spectrum, what was **J**?

HamLeT answers it by learning the inverse map from simulated pairs:

1. sample couplings **J**<sup>(n)</sup> from a stated range;
2. simulate each chain to get its spectrum *S*<sup>(n)</sup>;
3. fit an estimator *f* with *f*(*S*<sup>(n)</sup>) ≈ **J**<sup>(n)</sup>;
4. apply *f* to a measured spectrum.

Step 4 is only meaningful when the measurement resembles the training
distribution. That is why every artifact carries a contract — system, chain
length, view, bias window, observable, preprocessing — which HamLeT checks
and refuses rather than works around. Everything in §5 follows from this: the
accuracy reported for a model is accuracy **on simulated data**, and it
quantifies the inverse map, not whether the model Hamiltonian describes any
particular material.

## 2. The Hamiltonians

All chains are open (no periodic boundary) with *N* sites, each carrying spin
*S* (`site_spin`, default 1/2). Sums over *i* run along the chain.

### Bond-inhomogeneous Heisenberg

$$\hat{H} = \sum_{i=1}^{N-1} J_i\, \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_{i+1}$$

One independent isotropic coupling per bond. The inferred quantity is the
list *J*<sub>1</sub> … *J*<sub>*N*−1</sub>. This is the family of the
nanographene work, and the one the shipped local-window model was trained for.

### Homogeneous Heisenberg with longer range

$$\hat{H} = \sum_{d} J_d \sum_{i=1}^{N-d} \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_{i+d}$$

One isotropic coupling per interaction distance *d*, shared by the whole
chain. With *d* ∈ {1, 2} this is the *J*<sub>1</sub>–*J*<sub>2</sub> model.

### XXZ with isotropic *J*<sub>2</sub>, *J*<sub>3</sub>

$$\hat{H} = \sum_{i=1}^{N-1}\Big[ J_1^{xy}\big(\hat{S}^x_i \hat{S}^x_{i+1} + \hat{S}^y_i \hat{S}^y_{i+1}\big) + J^z\, \hat{S}^z_i \hat{S}^z_{i+1} \Big] + \sum_{d=2,3} J_d \sum_{i=1}^{N-d} \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_{i+d}$$

Nearest neighbours are anisotropic — *J*<sub>1</sub><sup>*xy*</sup> and
*J*<sup>*z*</sup> are independent — while the second- and third-neighbour
terms are isotropic. Parameter order: (*J*<sub>1</sub><sup>*xy*</sup>,
*J*<sub>2</sub>, *J*<sub>3</sub>, *J*<sup>*z*</sup>).

### Adding Dzyaloshinskii–Moriya

$$+ \sum_{i=1}^{N-1} D_z \big( \hat{S}^x_i \hat{S}^y_{i+1} - \hat{S}^y_i \hat{S}^x_{i+1} \big)$$

*D*<sub>*z*</sub> is the *z* component of a DM vector. §4 explains why this
term alone cannot be measured from the observable this package uses, and what
has to be added to the sample before it can.

### Symmetry-breaking terms

Anisotropic impurities at selected sites *a*, with an in-plane axis at angle
φ<sub>*a*</sub> from *x*:

$$\sum_{a} \Big[ D^a_\parallel \big(\hat{S}^z_a\big)^2 + E^a\Big(\cos 2\varphi_a \big[(\hat{S}^x_a)^2 - (\hat{S}^y_a)^2\big] + \sin 2\varphi_a \big[\hat{S}^x_a \hat{S}^y_a + \hat{S}^y_a \hat{S}^x_a\big]\Big) \Big]$$

and an optional uniform transverse field, which is a second, independent way
to break the same symmetry:

$$-\, B_x \sum_{i=1}^{N} \hat{S}^x_i$$

The field is along *x*, transverse to the DM vector along *z*. A field along
*z* would be invariant under the very rotation that removes a uniform
*D*<sub>*z*</sub> and would leave it exactly as unmeasurable as at zero field.

## 3. From Hamiltonian to a simulated dI/dV

**The observable.** HamLeT computes the on-site dynamical spin correlator at
each site *i*, a sum over spin components with weights *w*<sub>α</sub>:

$$S_{ii}(\omega) = \sum_{\alpha \in \{x,y,z\}} w_\alpha \, \big\langle \hat{S}^\alpha_i \,\big|\, \delta(\omega - \hat{H} + E_0) \,\big|\, \hat{S}^\alpha_i \big\rangle$$

evaluated with a Lorentzian broadening δ (`broadening_mev`). Two choices are
exposed: `Sz` uses *w* = (0, 0, 1), and `total_spin` uses *w* = (1, 1, 1).
The correlator is a *self*-correlator — the same operator on both sides — so
its spectral function is real; any imaginary part is numerical, which is what
the residue guard in §6 checks.

**The measurement model.** An STM at bias *V* collects every excitation below
*eV*, so the simulated dI/dV is the correlator integrated over bias:

$$\frac{\mathrm{d}I}{\mathrm{d}V}(V) \;\propto\; \int_0^{eV} S_{ii}(\omega)\, \mathrm{d}\omega$$

implemented as a cumulative trapezoid. **Each peak in the correlator becomes a
step in dI/dV** — which is the relationship the two left panels of the README
figure show, and which
[`scripts/make_readme_figures.py`](../scripts/make_readme_figures.py) verifies
numerically before drawing (the correlator integrates back to the stored
dI/dV to better than 10<sup>−6</sup> relative).

Set `output_quantity: spectral_function` to keep the correlator instead, or
`didv` (the default for datasets) for the integrated form.

**Views.** A *global* model reads the whole chain at once and is therefore
specific to one chain length. A *local* model reads a three-site sliding
window and returns the couplings on the bonds inside it, so it applies to any
chain long enough to contain a window. This is the practical difference
between the two shipped model kinds.

## 4. Why uniform DMI is invisible, and what fixes it

In a chain with U(1) symmetry about *z*, a uniform *D*<sub>*z*</sub> is not a
physical parameter of the spectrum. The site-dependent rotation

$$\hat{S}^\pm_j \rightarrow e^{\pm i j \theta} \hat{S}^\pm_j, \qquad \tan\theta = D_z / J_1^{xy}$$

absorbs it entirely, leaving an XXZ chain with a rescaled in-plane coupling
$\sqrt{(J_1^{xy})^2 + D_z^2}$. The on-site autocorrelator is invariant under
this rotation, so **no amount of data can separate *D*<sub>*z*</sub> from
*J*<sub>1</sub><sup>*xy*</sup>** — the two produce identical spectra. A model
trained to predict *D*<sub>*z*</sub> from such data would return a confident
number with no physical content.

HamLeT therefore does not offer an ordinary DMI model. It offers the
*impurity-assisted* route instead: add terms the rotation cannot absorb.

- **Axial impurity anisotropy** *D*<sub>∥</sub>(*S*<sup>*z*</sup>)² commutes
  with total *Ŝ*<sup>*z*</sup> and is invariant under the rotation. It cannot
  expose *D*<sub>*z*</sub>.
- **Transverse impurity anisotropy** *E* changes *S*<sup>*z*</sup> by two and
  breaks the U(1) symmetry. It can.
- **One transverse impurity is not enough.** The gauge rotation turns its
  in-plane axis by a single angle, and a global rotation about *z* undoes
  that. **Two impurities at distinct sites** are required: the rotation turns
  their axes by *different* angles, and no global rotation can undo both.
- **A transverse field** *B*<sub>*x*</sub> breaks the same symmetry
  independently, and can be combined with impurities.

The **DMI sample design** page screens candidate arrangements for exactly this
before anyone grows a sample: it reports whether the gauge rotation can undo
the proposed geometry, and how large a spectral difference the arrangement
produces. [`docs/dmi-experiment-spec.md`](dmi-experiment-spec.md) is the full
specification.

## 5. What a trained model's numbers mean

Training reserves whole simulated chains as a **test set**, used neither for
fitting nor for hyperparameter or ensemble selection, and reports on them
before the model may be applied to a measurement.

| quantity | definition | reads as |
|---|---|---|
| MAE | mean \|predicted − true\| | meV, lower better |
| RMSE | root mean square error | meV, penalises outliers |
| fidelity | \|Pearson *r*\| between predicted and true | 0 to 1, 1 perfect |
| skill | 1 − MAE<sub>model</sub> / MAE<sub>baseline</sub> | 0 = no better than guessing the training mean |

$$F = \frac{\big| \langle (J_{\text{pred}} - \langle J_{\text{pred}} \rangle)(J_{\text{true}} - \langle J_{\text{true}} \rangle) \rangle \big|}{\sigma_{\text{pred}}\, \sigma_{\text{true}}}$$

Each is reported overall **and per learned parameter**, because a model can
recover one coupling well and another not at all, and an aggregate hides it.

Read fidelity next to the MAE, never instead of it: being a correlation, it is
blind to a constant offset and to a scale factor, so a model can score highly
on it and still be wrong by several meV. Skill answers the complementary
question — whether the model beats the trivial estimator that ignores the
spectrum and always returns the training mean.

**Ensemble spread** at inference measures disagreement between trained members.
It is not a calibrated uncertainty and should not be read as an error bar.

## 6. Numerical method and its limits

**Solver.** Exact diagonalisation while the Hilbert space is affordable
(≤ 2048 basis states — an eight-site spin-1/2 chain is 256), DMRG above it.
`dynamics_mode: auto` picks; `ED` or `DMRG` overrides. ED is exact up to
floating point; DMRG is approximate and its accuracy follows the bond
dimension. The choice is part of the recipe fingerprint, because the two give
different numbers.

**The imaginary-residue guard.** The spectral function of a self-correlator is
real, so HamLeT refuses a result whose imaginary part is too large to be
numerical noise. The threshold follows the solver — 10<sup>−6</sup> for ED,
10<sup>−3</sup> for DMRG — because an MPS truncation leaves a residue around
10<sup>−6</sup>–10<sup>−5</sup>, and judging it by the exact solver's standard
rejects perfectly sound chains.

**Determinism.** Chain seeds are derived from a seed sequence by index, so a
given recipe produces a bit-identical dataset regardless of how many cores or
cluster tasks generated it, and in what order they finished.

## 7. Assumptions worth stating plainly

- The chain is **open**, uniform in spin magnitude unless impurities say
  otherwise, and at **zero temperature** — the correlator is evaluated against
  the ground state.
- dI/dV is taken as **proportional** to the integrated correlator. Tip
  transfer functions, tunnelling matrix elements, and elastic background are
  not modelled; preprocessing and the augmentation calibration exist to absorb
  what they do to the lineshape, not to derive them.
- Accuracy figures describe **simulated** spectra from the same generator that
  produced the training set.
- A measurement must satisfy the artifact contract before its predictions are
  physically interpretable. HamLeT reports a mismatch instead of padding
  spectra or silently changing the cutoff.

## Further reading

The papers this package implements, and the model cards recording what each
shipped artifact was trained on, are listed in the
[README](../README.md#work-this-package-builds-on). Cite them if you use the
impurity route or the uniform chain families; the estimators you train are
your own.
