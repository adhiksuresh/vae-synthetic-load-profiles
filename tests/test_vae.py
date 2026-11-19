import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vaeload import data, evaluate, vae  # noqa: E402


def test_profiles_are_positive_and_shaped():
    x, meta = data.generate_profiles(n_households=5, n_days=7)
    assert x.shape == (35, 48)
    assert (x > 0).all()
    assert meta["weekend"].sum() == 5 * 2


def test_split_has_no_household_overlap():
    x, meta = data.generate_profiles(n_households=20, n_days=3)
    tr, va, te = data.split_by_household(x, meta)
    assert len(tr) + len(va) + len(te) == len(x)
    rows = lambda a: {tuple(r) for r in np.round(a, 8)}  # noqa: E731
    assert not rows(tr) & rows(te)


def test_analytic_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    m = vae.VAE(d_in=10, hidden=7, latent=3, s2=0.2, beta=0.7, seed=1)
    x = rng.standard_normal((6, 10))
    eps = rng.standard_normal((6, 3))
    assert vae.gradient_check(m, x, eps, n_checks=20) < 1e-6


def test_training_reduces_loss():
    x, _ = data.generate_profiles(n_households=20, n_days=10)
    sc = evaluate.LogStandardiser().fit(x)
    m = vae.VAE(latent=4, seed=0)
    hist = vae.train(m, sc.transform(x), epochs=15)
    assert hist[-1, 0] < hist[0, 0]


def test_log_standardiser_round_trip():
    x, _ = data.generate_profiles(n_households=3, n_days=3)
    sc = evaluate.LogStandardiser().fit(x)
    assert np.allclose(sc.inverse(sc.transform(x)), x)
