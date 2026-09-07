# Measuring DMI: designing a sample that can

DMI is the one coupling this package cannot simply be pointed at. A DM vector
along `z` is *exactly* unidentifiable in a conventional chain, so measuring it
is a question of sample design before it is a question of analysis.

This guide is therefore organised as:

1. **The rule** — one criterion that decides whether any design can work.
2. **Screening your own design** — the package computes this for your chain,
   your length, your impurities, before you generate anything.
3. **A worked example** — one configuration taken end to end, as a reference
   point rather than a prescription.
4. **Precision requirements** — how exactly the impurities must be known, and
   what it costs to be wrong.

If you already have impurities characterised, go to
[§4 Screening](#4-screening-your-own-design) and put your measured values in.
If you are choosing what to build, screening is also how you compare candidates
cheaply — two simulations each, instead of a training set each.

Every number here was measured on simulated exact-diagonalisation chains with
this package, not estimated.

---

## 1. The problem: DMI is invisible in a normal chain

For a chain with a Dzyaloshinskii–Moriya vector along `z`, the transverse
exchange is a complex hopping `t = (J1_xy - i D_z) / 2`. The site-dependent
rotation

```
U = prod_j exp(-i theta_j S^z_j)
```

removes that phase. On an **open** chain the phases can be chosen bond by bond,
for arbitrary per-bond values and any spin magnitude. What is left is a plain
XXZ chain with `J'_xy = sqrt(J1_xy^2 + D_z^2)`.

The consequence is exact, not statistical:

> **`D_z` is unidentifiable in any Hamiltonian that conserves total `S^z`.**

Measured: two chains sharing `sqrt(J1_xy^2 + D_z^2)` but splitting it
differently between exchange and DMI produce dI/dV maps agreeing to **1 part in
10^13** — machine precision, with a deliberately sensitive control confirming
the test could see a 4.4% difference. Only the combination
`sqrt(J1_xy^2 + D_z^2)` is measurable, never `D_z` alone.

So a conventional spin chain cannot yield `D_z`, and no amount of data,
resolution or model sophistication changes that. Something must break the U(1)
symmetry about the DM axis.

## 2. What does not work

These were each measured and each leaves the degeneracy at machine precision
(~10^-14 relative):

| approach | why it fails |
|---|---|
| A longer or better-resolved measurement | the degeneracy is exact, not a resolution limit |
| Bond-disordered `D_z` along the chain | still removable bond by bond on an open chain |
| A vacancy, or cutting the chain | produces shorter open chains, still gaugeable |
| Substituting a **different spin** (S=1, S=5/2 …) | `S^±` transforms identically at every spin magnitude |
| Easy-axis anisotropy `D(S^z)^2` on an impurity | commutes with total `S^z` |
| **One** impurity with transverse anisotropy | see below — a single in-plane axis is undone by a global rotation |
| A field along `z` | itself invariant under the rotation that hides `D_z` |
| Spin-polarised tip on a symmetric chain | each on-site component is separately invariant when U(1) holds |

The single-impurity case is the subtle one and cost a wrong prediction before it
was measured. Transverse anisotropy `E((S^x)^2-(S^y)^2)` genuinely breaks U(1),
but the gauge rotates that one impurity's in-plane axis by a single angle
`theta_s`, and the global rotation `R_z(-theta_s)` rotates it back while leaving
the exchange, `Jz` and `D_z` untouched. **Relative** angles between two or more
impurities are what survive.

## 3. What works

| recipe | imprint on a gauge pair | notes |
|---|---|---|
| 2 impurities, 1 site apart | 8.3e-03 | too weak |
| 2 impurities, 3 sites apart | 5.7e-02 | marginal |
| 2 impurities, 5 sites apart | 8.0e-02 | marginal |
| transverse field `B_x` = 1 meV | 1.18e-01 | ~8.6 T; trained to only 0.19 skill, and **flat** with more data |
| 2 impurities, `E` = 2 meV | 1.36e-01 | |
| **3 impurities, `E` = 2 meV, no field** | **2.29e-01** | **recommended** |

Two findings worth stating because they are counter-intuitive:

- **A field does not help once you have impurities.** Combining them makes
  things *worse* (3 impurities alone 2.29e-01; with a 1 meV field 1.41e-01).
  The mechanisms interfere rather than add. Do not apply a field.
- **`E` is non-monotonic.** `E` = 3 meV is worse than `E` = 2 meV.

The field-only route is included for completeness because it needs no
impurities, but it is not recommended: it plateaued at 0.19 `D_z` skill and its
learning curve was flat (0.24 → 0.21 → 0.24 as chains tripled), so more
measurement time cannot rescue it. It also needs `B/J` of order 10%, which is
~8.6 T for a few-meV chain and an unreachable ~43 T for a 35 meV chain.

## 4. Screening your own design

Do not take the configuration in §5 as a requirement. Which impurity
arrangement is best depends on your chain length, your exchange scale, and what
species you can actually place — so the package computes it for your case. It
simulates a **gauge pair**: two chains sharing `sqrt(J1_xy^2 + D_z^2)` that
differ only in how it splits between exchange and DMI. A design that cannot
separate that pair cannot yield `D_z`, whatever model is trained on it.

Two simulations per candidate, against thousands of chains for a training set.

### From a configuration file

No Python needed. Declare the chain, the measurement, and the candidates:

```bash
cp examples/dmi_screening.yaml my_screening.yaml
hamlet screen-dmi my_screening.yaml --json screening.json
```

```yaml
chain:
  n_sites: 8
  j_eff_mev: 5.0      # the sqrt(J1_xy^2 + D_z^2) a zero-field measurement gives
  d_z_mev: 1.5        # the DMI you are trying to resolve
  jz_mev: 5.5

sweep:                            # use this when choosing what to build
  sites: [[3], [1, 6], [1, 4, 6]]
  transverse_mev: [1.0, 2.0]
  spin: S=1
  axial_mev: 0.0                  # your measured axial anisotropy

# candidates:                     # use this when the impurities are measured
#   - label: measured Fe/Fe/Mn chain
#     impurities:
#       - {site: 1, spin: "S=1",   transverse_mev: 1.8, axial_mev: 1.4}
#       - {site: 6, spin: "S=3/2", transverse_mev: 2.2, axial_mev: 2.1}
```

Real output from the shipped example:

```
                            design    imprint  verdict
--------------------------------------------------------------
           3 imp at [1, 4, 6], E=2  2.286e-01  strong
              2 imp at [1, 6], E=2  1.358e-01  promising
           3 imp at [1, 4, 6], E=1  9.010e-02  marginal
              2 imp at [1, 6], E=1  8.004e-02  marginal
                 1 imp at [3], E=1  0.000e+00  hidden
                 1 imp at [3], E=2  0.000e+00  hidden
```

Note what that shows beyond the ranking: `E` is **non-monotonic and
interacts with impurity count** — three impurities at `E`=1 (9.0e-02) are worse
than two at `E`=2 (1.36e-01). There is no rule of thumb to substitute for
screening your own case.

Unknown fields in the configuration are rejected rather than ignored, and a
configuration in which nothing can break the symmetry exits non-zero without
simulating anything.

### From Python

```python
from hamlet import DmiDesign, screen_dmi_designs, transverse_impurities
from hamlet import format_screening_table
from hamlet.simulation import DmrgpySimulator, SpectroscopyProtocol

protocol = SpectroscopyProtocol.uniform(
    (0.0, 20.0), points=81, broadening_mev=0.25,
    observable="total_spin", observable_weights=(1.0, 1.0, 1.0),
    output_quantity="didv",
)

# Your chain: length, exchange scale, and the DMI you are trying to resolve.
# Then the candidate impurity arrangements you could actually build.
candidates = [
    DmiDesign(n_sites=10, j_eff_mev=4.0, d_z_mev=1.2, jz_mev=4.0,
              impurities=transverse_impurities(sites, e_mev, spin="S=1",
                                               axial_mev=1.4),
              label=f"{len(sites)} imp at {sites}, E={e_mev}")
    for sites in ([2, 7], [1, 5, 8], [0, 3, 6, 9])
    for e_mev in (1.0, 2.0)
]

results = screen_dmi_designs(candidates, DmrgpySimulator(dynamics_mode="ED"), protocol)
print(format_screening_table(results))
```

Each result carries an `imprint` and a `verdict` — `hidden`, `too weak`,
`marginal`, `promising`, `strong`. Those are not invented thresholds: they are
anchored to the `D_z` skill that models trained on such designs actually
reached, which `format_screening_table` prints alongside the results so you can
judge for yourself.

Designs with no U(1)-breaking mechanism are recognised from the symmetry rule
and reported as `hidden` **without being simulated**, so sweeping impurity
counts and positions is cheap. Pass `skip_symmetric=False` to verify that
prediction against the simulator instead of trusting it.

`DmiDesign.breaks_symmetry` gives the free structural answer on its own, and
`measure_dmi_imprint` screens a single design.

### Once you have a design

Put the same impurities into a project config and the workflow is ordinary:

```bash
cp examples/heisenberg_xxz_dmi_impurities_l8.yaml my_chain.yaml
# edit n_sites, coupling_ranges_mev, and the impurities block
hamlet generate my_chain.yaml --dry-run   # cost first
hamlet run my_chain.yaml
```

The `impurities` block takes any number of impurities at arbitrary distinct
sites, each with its own `spin`, `axial_mev`, `transverse_mev` and
`transverse_angle_rad`, so a measured configuration maps onto it directly. Any
chain length the simulator can handle is allowed; cost grows steeply with it,
and with substituted spins above 1/2.

## 5. A worked example

This is the configuration taken end to end in this package, and the one the
published reference artifact uses. It is a **reference point, not a
requirement** — screen your own candidates in §4 rather than adopting it
blindly, especially if your exchange scale or available species differ.

- **Host chain:** 8 spin-1/2 sites, open (not a ring).
- **Exchange:** weakly coupled, `J1_xy` and `Jz` in **2–6 meV**. This matters:
  identifiability is governed by ratios to `J`, so a 35 meV chain would need a
  proportionally larger perturbation.
- **Impurities:** **three** substituted sites carrying transverse magnetic
  anisotropy, at **sites 1, 4 and 6** (zero-based, so the 2nd, 5th and 7th
  atoms).
- **Impurity spin:** **S ≥ 1**. A spin-1/2 impurity cannot work — single-ion
  anisotropy is a constant for S=1/2.
- **Transverse anisotropy:** `E` ≈ **2 meV** per impurity.
- **Magnetic field:** **none.**

## 6. Precision requirements

A model is trained for one exact impurity configuration. The cost of a mismatch
was measured by running the trained model on 200 fresh chains per perturbed
condition:

| what differs | `D_z` MAE [meV] | `D_z` skill |
|---|---|---|
| nothing (reference) | 0.263 | **0.50** |
| `E` off by 10% | 0.51 | 0.02 |
| `E` off by 20% | 0.89–0.95 | −0.7 to −0.8 |
| axial `D` = 1 meV present but unmodelled | 0.544 | −0.04 |
| axial `D` = 2 meV present but unmodelled | 0.798 | −0.52 |
| one impurity one lattice site off | **11.19** | **−20.3** |

Reading that as requirements:

1. **`E` must be characterised to a few percent.** A 10% error removes all DMI
   skill; 20% is worse than ignoring the data and quoting the training average.
2. **Impurity positions must be exact and known.** One site off gives an 11 meV
   error on a quantity whose entire range is 0.3–2.5 meV.
3. **Axial anisotropy `D` must also be measured and declared**, even though it
   cannot expose `D_z` by itself. Real adatoms have substantial `D`, and 1 meV
   of unmodelled `D` is already enough to reach zero skill.

This is why the practical deliverable is a **method, not a single model**:
characterise the impurities, then generate and train for that specific chain.

### Why you cannot rely on the software to notice

Out-of-distribution detection **does not catch** any of this. The mismatched
chains score as in-distribution (6–7% flagged, against 5% for the reference),
and even the catastrophic one-site shift has a median novelty score below the
95th-percentile threshold. The spectra look entirely ordinary; only the mapping
from spectra to couplings has changed.

A wrong answer therefore arrives with confident-looking inputs and no warning.
`hamlet advise` compares declared impurity conditions against the artifact and
refuses reuse on any difference — that check is the only defence, and it depends
on the conditions you declare being true.

## 7. Pre-characterisation checklist

Before assembling the chain, measure each impurity species **in isolation**:

- [ ] axial anisotropy `D` (meV)
- [ ] transverse anisotropy `E` (meV), to a few percent
- [ ] spin magnitude `S` (must be ≥ 1)
- [ ] in-plane orientation of the anisotropy axes, if it differs between sites

And record for the assembled chain:

- [ ] chain length (the model is length-specific; L=8 here)
- [ ] exact impurity site indices
- [ ] confirmation that no external field was applied

## 8. Measurement protocol

| quantity | requirement |
|---|---|
| observable | dI/dV at **every** site of the chain |
| tip | ordinary unpolarised tip (no spin polarisation needed) |
| bias window | **0 to 20 meV**, starting at zero |
| bias points | **≥ 81** across that window |
| energy resolution | **≤ 0.25 meV** effective broadening |
| completeness | no missing sites and no gaps; the pipeline refuses partial maps |

The bias window and resolution are matched to a 2–6 meV chain, where the
excitations sit at a few meV. A stiffer chain needs both rescaled.

## 9. What you get, and what you do not

Recovered on 3000 simulated L=8 chains, five held-out splits, all resolved
against a training-mean baseline:

| parameter | test MAE [meV] | skill |
|---|---|---|
| `J1_xy` | 0.169 | 0.83 |
| `J2` | 0.093 | 0.88 |
| `J3` | 0.136 | 0.73 |
| `Jz` | 0.235 | 0.77 |
| `D_z` magnitude | 0.258 | 0.54 |

Not recovered:

- **The sign of `D_z`.** Only its magnitude is a target.
- **`D_z` when `D_z / J1_xy` is small.** Skill falls to ~0.06 below a ratio of
  0.1, against ~0.3 above 0.2, because the recoverable quantity is really the
  angle `arctan(D_z / J1_xy)`. Chains with negligible DMI cannot report their
  negligible DMI accurately.
- **Anything at a different chain length.** The global view takes the whole
  site-by-bias map as one input, so L=8 means L=8.

Accuracy figures are against simulated spectra from the same generator that
produced the training data. They quantify the inverse problem, not whether the
model Hamiltonian describes any particular material, and the ensemble spread
reported at inference measures agreement between seeds rather than distance
from truth.

See [user-guide.md](user-guide.md) for the general workflow.
