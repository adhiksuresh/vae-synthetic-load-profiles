"""Full experiment: data -> gradient check -> model selection -> training -> evaluation -> figures.

Usage:
    python scripts/run_experiment.py

Households are split three ways (train / validation / test). Hyperparameters
are chosen on validation households only; the test households are used once,
at the end.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vaeload import data, evaluate, vae  # noqa: E402

FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
SEED = 0
GRID = [(lat, s2) for lat in (4, 8, 16) for s2 in (0.05, 0.1, 0.2)]
EPOCHS = 150
HOURS = np.arange(48) / 2


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- data: 500 households x 60 days, split by household 60/20/20 ----
    X, meta = data.generate_profiles(n_households=500, n_days=60, seed=SEED)
    train, val, test = data.split_by_household(X, meta, seed=SEED)
    print(f"profiles: train {len(train)}, validation {len(val)}, test {len(test)}")

    scaler = evaluate.LogStandardiser().fit(train)
    x_train = scaler.transform(train)

    # ---- gradient check ----
    probe = vae.VAE(latent=8, s2=0.1, seed=SEED)
    eps = rng.standard_normal((32, 8))
    grad_err = vae.gradient_check(probe, x_train[:32], eps, n_checks=10)
    print(f"gradient check: max relative error {grad_err:.2e}")

    # ---- model selection on validation households (criterion: MMD) ----
    rows = []
    for latent, s2 in GRID:
        m = vae.VAE(latent=latent, s2=s2, seed=SEED)
        vae.train(m, x_train, epochs=EPOCHS, seed=SEED)
        gen = scaler.inverse(m.sample(len(val), np.random.default_rng(1)))
        r = evaluate.fidelity(val, gen, np.random.default_rng(2))
        rows.append({"latent": latent, "s2": s2, **r})
        print(f"  latent={latent:2d} s2={s2:.2f}  val MMD={r['MMD']:.4f}  peak KS={r['peak_time_KS_stat']:.3f}")
    sel = pd.DataFrame(rows)
    sel.to_csv(TAB / "model_selection_validation.csv", index=False)
    best = sel.sort_values("MMD").iloc[0]
    latent, s2 = int(best["latent"]), float(best["s2"])
    print(f"selected: latent={latent}, s2={s2}")

    # ---- final model ----
    model = vae.VAE(latent=latent, s2=s2, seed=SEED)
    hist = vae.train(model, x_train, epochs=EPOCHS, seed=SEED)

    # ---- test evaluation (used once) ----
    g = np.random.default_rng(3)
    n = len(test)
    candidates = {
        "Real (other households, reference)": train[g.choice(len(train), n, replace=False)],
        "VAE": scaler.inverse(model.sample(n, g)),
        "Gaussian baseline": scaler.inverse(evaluate.gaussian_baseline(x_train, n, g)),
    }
    results = pd.DataFrame(
        {name: evaluate.fidelity(test, gen, np.random.default_rng(4)) for name, gen in candidates.items()}
    ).T
    results.index.name = "generator"
    results.to_csv(TAB / "test_fidelity.csv")
    print(results[["daily_energy_KS_stat", "peak_time_KS_stat", "correlation_matrix_MAE", "MMD"]])

    (TAB / "run_info.json").write_text(json.dumps({
        "gradient_check_max_relative_error": float(grad_err),
        "selected_latent": latent, "selected_s2": s2, "epochs": EPOCHS,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }, indent=2))

    _figures(test, candidates, hist)


def _figures(test, candidates, hist):
    vae_gen, gauss = candidates["VAE"], candidates["Gaussian baseline"]
    g = np.random.default_rng(5)

    # example profiles
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (title, arr) in zip(axes, [("Real (test households)", test), ("VAE samples", vae_gen), ("Gaussian baseline samples", gauss)]):
        for i in g.choice(len(arr), 8, replace=False):
            ax.plot(HOURS, arr[i], lw=1, alpha=0.8)
        ax.set(title=title, xlabel="Hour of day", xticks=range(0, 25, 6))
    axes[0].set_ylabel("kWh per half hour")
    fig.tight_layout(); fig.savefig(FIG / "example_profiles.png", dpi=150); plt.close(fig)

    # mean and spread
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, arr, c in [("Real (test)", test, "black"), ("VAE", vae_gen, "#E45756"), ("Gaussian", gauss, "#4C78A8")]:
        ax.plot(HOURS, arr.mean(0), color=c, lw=2, label=f"{name} mean")
        ax.fill_between(HOURS, np.percentile(arr, 10, 0), np.percentile(arr, 90, 0), color=c, alpha=0.12)
    ax.set(title="Mean profile and 10-90% band", xlabel="Hour of day", ylabel="kWh per half hour", xticks=range(0, 25, 3))
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "mean_and_spread.png", dpi=150); plt.close(fig)

    # distributions: daily energy and peak time
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins_e = np.linspace(0, np.percentile(test.sum(1), 99.5), 40)
    for name, arr, c in [("Real (test)", test, "black"), ("VAE", vae_gen, "#E45756"), ("Gaussian", gauss, "#4C78A8")]:
        axes[0].hist(arr.sum(1), bins=bins_e, histtype="step", density=True, color=c, lw=1.6, label=name)
        axes[1].hist(arr.argmax(1) / 2, bins=np.arange(0, 24.5, 0.5), histtype="step", density=True, color=c, lw=1.6, label=name)
    axes[0].set(title="Daily energy", xlabel="kWh per day", ylabel="Density")
    axes[1].set(title="Time of daily peak", xlabel="Hour of day")
    axes[1].legend(); fig.tight_layout(); fig.savefig(FIG / "distributions.png", dpi=150); plt.close(fig)

    # training curve
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(hist[:, 1], label="Reconstruction"); ax.plot(hist[:, 2], label="KL")
    ax.set(title="Training loss terms (per profile)", xlabel="Epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "training_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
