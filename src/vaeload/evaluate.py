"""Statistical fidelity checks for generated load profiles."""

import numpy as np
from scipy import stats


class LogStandardiser:
    """log1p transform then per-time-step standardisation.

    Working in log space guarantees generated loads are positive after the
    inverse transform.
    """

    def fit(self, x):
        lx = np.log1p(x)
        self.mean, self.std = lx.mean(0), lx.std(0) + 1e-8
        return self

    def transform(self, x):
        return (np.log1p(x) - self.mean) / self.std

    def inverse(self, z):
        return np.expm1(z * self.std + self.mean)


def gaussian_baseline(x_train_std, n, rng):
    """Baseline generator: multivariate normal fitted to the (transformed) training data."""
    mean = x_train_std.mean(0)
    cov = np.cov(x_train_std, rowvar=False)
    return rng.multivariate_normal(mean, cov, size=n)


def _mmd_rbf(a, b, gamma=None):
    """Maximum mean discrepancy with an RBF kernel (smaller = more similar)."""
    xy = np.vstack([a, b])
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    if gamma is None:
        gamma = 1.0 / np.median(d2[d2 > 0])
    k = np.exp(-gamma * d2)
    n = len(a)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def fidelity(real, fake, rng, mmd_n=400):
    """Compare real and generated profiles (both in kWh per half hour)."""
    e_real, e_fake = real.sum(1), fake.sum(1)
    pk_real, pk_fake = real.argmax(1) / 2, fake.argmax(1) / 2
    ks_energy = stats.ks_2samp(e_real, e_fake)
    ks_peak = stats.ks_2samp(pk_real, pk_fake)
    corr_real = np.corrcoef(real, rowvar=False)
    corr_fake = np.corrcoef(fake, rowvar=False)
    ia = rng.choice(len(real), min(mmd_n, len(real)), replace=False)
    ib = rng.choice(len(fake), min(mmd_n, len(fake)), replace=False)
    return {
        "daily_energy_mean_real": round(float(e_real.mean()), 3),
        "daily_energy_mean_gen": round(float(e_fake.mean()), 3),
        "daily_energy_KS_stat": round(float(ks_energy.statistic), 3),
        "daily_energy_KS_p": round(float(ks_energy.pvalue), 3),
        "peak_time_KS_stat": round(float(ks_peak.statistic), 3),
        "mean_profile_RMSE": round(float(np.sqrt(((real.mean(0) - fake.mean(0)) ** 2).mean())), 4),
        "std_profile_RMSE": round(float(np.sqrt(((real.std(0) - fake.std(0)) ** 2).mean())), 4),
        "correlation_matrix_MAE": round(float(np.abs(corr_real - corr_fake).mean()), 4),
        "MMD": round(float(_mmd_rbf(real[ia], fake[ib])), 4),
        "negative_values_pct": round(float(100 * (fake < 0).mean()), 3),
    }
