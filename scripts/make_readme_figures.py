"""Build the README pipeline figure from real package data.

Nothing here is drawn by hand: the heatmap and the dI/dV traces are one
simulated chain out of the dataset shipped for the quickstart, and the
recovered couplings come from a ridge model trained on the others.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from hamlet.data import SpectroscopyDataset
from hamlet.training import prepare_training_dataset, train_supervised
from hamlet.training.preprocessing import TrainingPreprocessingConfig

INK, MUTED, ACCENT, LINE = "#202d38", "#566674", "#166b79", "#dce3e7"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": "white",
})

root = Path(__file__).resolve().parents[1]
data = SpectroscopyDataset.load(root / "examples/l8_demo/homogeneous_xxz_l8_ed.npz")
bias = np.asarray(data.bias_mev)

# Hold one chain out, train on the rest, then infer it back.
held = 0
keep = np.arange(1, data.n_samples)
train_set = SpectroscopyDataset(
    spectra=data.spectra[keep], targets_mev=data.targets_mev[keep],
    bias_mev=data.bias_mev, target_names=data.target_names,
    system_type=data.system_type, metadata=dict(data.metadata),
)
config = TrainingPreprocessingConfig(bias_cutoff_mev=float(bias.max()), output_points=len(bias))
prepared = prepare_training_dataset(train_set, config)
run = train_supervised(prepared, view="global", model="ridge", preset="quick",
                       model_options={"alpha": 0.1}, verbose=0)

# Preprocess the held-out chain exactly as the training data was.
one = SpectroscopyDataset(
    spectra=data.spectra[[held]], targets_mev=data.targets_mev[[held]],
    bias_mev=data.bias_mev, target_names=data.target_names,
    system_type=data.system_type, metadata=dict(data.metadata),
)
prepared_one = prepare_training_dataset(one, config)
from hamlet.data import as_supervised
sup = as_supervised(prepared_one.dataset, "global")
predicted = run.predict(sup.inputs)[0]
truth = data.targets_mev[held]
sample = data.spectra[held]
print("true      :", np.round(truth, 2))
print("recovered :", np.round(predicted, 2))

# --- the figure ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)

# 1. what the simulator produces: the site-resolved dynamical correlator.
ax = axes[0]
mesh = ax.pcolormesh(bias, np.arange(sample.shape[0]), sample, cmap="magma",
                     shading="gouraud")
ax.set_title("1.  Simulate", loc="left")
ax.set_xlabel("bias [meV]"); ax.set_ylabel("site along the chain")
ax.set_yticks(range(sample.shape[0]))
cb = fig.colorbar(mesh, ax=ax, pad=0.02); cb.set_label("spectral weight", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.text(0.5, -0.34, "a known Hamiltonian $\\rightarrow$ its dynamical correlator",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)

# 2. the same thing as the experiment sees it: one dI/dV curve per site.
ax = axes[1]
offset = 0.55 * np.nanmax(sample)
for site in range(sample.shape[0]):
    ax.plot(bias, sample[site] + site * offset, color=ACCENT, lw=1.4)
    ax.text(bias[-1], site * offset + 0.12 * offset, f" {site}", fontsize=7, color=MUTED,
            va="center", ha="left", clip_on=False)
ax.set_title("2.  Measure", loc="left")
ax.set_xlabel("bias [meV]"); ax.set_ylabel("d$I$/d$V$  (offset per site)")
ax.set_yticks([]); ax.set_xlim(bias[0], bias[-1] * 1.04)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.text(0.5, -0.34, "one spectrum per site, the STM measurement",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)

# 3. the answer: couplings recovered from that spectrum alone.
ax = axes[2]
names = [n.replace("_", "$_{") + "}$" if "_" in n else n for n in data.target_names]
y = np.arange(len(truth))[::-1]
ax.barh(y + 0.19, truth, height=0.36, color="#c3ced6", label="true")
ax.barh(y - 0.19, predicted, height=0.36, color=ACCENT, label="inferred")
ax.axvline(0, color=LINE, lw=0.8, zorder=0)
# Values in a column clear of the longest bar, so nothing overlaps whichever
# way the couplings happen to fall.
label_x = max(truth.max(), predicted.max()) * 1.08
for i, (t_, p_) in enumerate(zip(truth, predicted)):
    ax.text(label_x, y[i], f"{p_:.2f}", va="center", ha="left", fontsize=8.5,
            color=INK, fontweight="bold")
ax.set_xlim(min(0, truth.min() * 1.3), label_x * 1.24)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_ylim(-0.7, len(truth) - 0.3)
ax.set_title("3.  Infer", loc="left")
ax.set_xlabel("coupling [meV]")
# Above the bars, where no data can reach it.
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 0.99), handlelength=1.2, columnspacing=1.2)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.text(0.5, -0.34, "the Hamiltonian, read back out of the spectrum",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)

out = root / "assets/figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "pipeline.png", dpi=155, bbox_inches="tight", facecolor="white")
print("wrote", out / "pipeline.png")
