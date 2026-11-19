"""A variational autoencoder written from first principles in NumPy.

Architecture (D = 48 inputs, H hidden units, K latent dimensions):

    encoder:  x -> tanh(x W1 + b1) = h1 -> mu = h1 Wm + bm,  logvar = h1 Wv + bv
    sample:   z = mu + exp(logvar / 2) * eps,   eps ~ N(0, I)
    decoder:  z -> tanh(z W3 + b3) = h2 -> x_hat = h2 W4 + b4

Loss per example (averaged over the batch):

    L = ||x - x_hat||^2 / (2 s2)  +  beta * KL( N(mu, diag(exp(logvar))) || N(0, I) )
    KL = -1/2 * sum(1 + logvar - mu^2 - exp(logvar))

``s2`` is a fixed observation-noise variance. All gradients are derived by
hand in :meth:`VAE.backward` and verified numerically in ``tests/``.
"""

import numpy as np

PARAM_NAMES = ["W1", "b1", "Wm", "bm", "Wv", "bv", "W3", "b3", "W4", "b4"]


class VAE:
    def __init__(self, d_in=48, hidden=64, latent=4, s2=0.1, beta=1.0, seed=0):
        rng = np.random.default_rng(seed)

        def glorot(n_in, n_out):
            return rng.normal(0, np.sqrt(2.0 / (n_in + n_out)), size=(n_in, n_out))

        self.p = {
            "W1": glorot(d_in, hidden), "b1": np.zeros(hidden),
            "Wm": glorot(hidden, latent), "bm": np.zeros(latent),
            "Wv": glorot(hidden, latent) * 0.1, "bv": np.zeros(latent),
            "W3": glorot(latent, hidden), "b3": np.zeros(hidden),
            "W4": glorot(hidden, d_in), "b4": np.zeros(d_in),
        }
        self.latent = latent
        self.s2 = s2
        self.beta = beta

    # ----- forward -------------------------------------------------------
    def encode(self, x):
        h1 = np.tanh(x @ self.p["W1"] + self.p["b1"])
        mu = h1 @ self.p["Wm"] + self.p["bm"]
        logvar = h1 @ self.p["Wv"] + self.p["bv"]
        return h1, mu, logvar

    def decode(self, z):
        h2 = np.tanh(z @ self.p["W3"] + self.p["b3"])
        return h2, h2 @ self.p["W4"] + self.p["b4"]

    def forward(self, x, eps):
        """Return loss terms and a cache for backward. ``eps`` is passed in so
        the forward pass is deterministic (needed for gradient checking)."""
        h1, mu, logvar = self.encode(x)
        std = np.exp(0.5 * logvar)
        z = mu + std * eps
        h2, x_hat = self.decode(z)

        n = x.shape[0]
        recon = np.sum((x - x_hat) ** 2) / (2 * self.s2) / n
        kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / n
        cache = dict(x=x, h1=h1, mu=mu, logvar=logvar, std=std, eps=eps, z=z, h2=h2, x_hat=x_hat)
        return recon + self.beta * kl, recon, kl, cache

    # ----- backward ------------------------------------------------------
    def backward(self, cache):
        c, p, n = cache, self.p, cache["x"].shape[0]
        g = {}

        # reconstruction term
        dx_hat = (c["x_hat"] - c["x"]) / self.s2 / n
        g["W4"] = c["h2"].T @ dx_hat
        g["b4"] = dx_hat.sum(0)
        da2 = (dx_hat @ p["W4"].T) * (1 - c["h2"] ** 2)
        g["W3"] = c["z"].T @ da2
        g["b3"] = da2.sum(0)
        dz = da2 @ p["W3"].T

        # reparameterisation + KL term
        dmu = dz + self.beta * c["mu"] / n
        dlogvar = dz * c["eps"] * 0.5 * c["std"] + self.beta * 0.5 * (np.exp(c["logvar"]) - 1) / n

        g["Wm"] = c["h1"].T @ dmu
        g["bm"] = dmu.sum(0)
        g["Wv"] = c["h1"].T @ dlogvar
        g["bv"] = dlogvar.sum(0)
        da1 = (dmu @ p["Wm"].T + dlogvar @ p["Wv"].T) * (1 - c["h1"] ** 2)
        g["W1"] = c["x"].T @ da1
        g["b1"] = da1.sum(0)
        return g

    # ----- generation ----------------------------------------------------
    def sample(self, n, rng, add_noise=True):
        """Draw new examples: z ~ N(0, I), decode, optionally add observation noise."""
        z = rng.standard_normal((n, self.latent))
        _, x_hat = self.decode(z)
        if add_noise:
            x_hat = x_hat + rng.normal(0, np.sqrt(self.s2), x_hat.shape)
        return x_hat


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gradient_check(model: VAE, x, eps, h=1e-5, n_checks=5, seed=0):
    """Compare analytic gradients with central finite differences.

    Returns the largest relative error over ``n_checks`` random entries of
    every parameter array.
    """
    rng = np.random.default_rng(seed)
    _, _, _, cache = model.forward(x, eps)
    grads = model.backward(cache)
    worst = 0.0
    for name in PARAM_NAMES:
        arr = model.p[name]
        for _ in range(n_checks):
            idx = tuple(rng.integers(s) for s in arr.shape)
            old = arr[idx]
            arr[idx] = old + h
            lp = model.forward(x, eps)[0]
            arr[idx] = old - h
            lm = model.forward(x, eps)[0]
            arr[idx] = old
            num = (lp - lm) / (2 * h)
            ana = grads[name][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


def train(model: VAE, x_train, epochs=200, batch=128, lr=2e-3, seed=0, verbose=False):
    """Mini-batch training with Adam. Returns per-epoch (loss, recon, kl)."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    history = []
    n = len(x_train)
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = np.zeros(3)
        for i in range(0, n, batch):
            xb = x_train[order[i:i + batch]]
            eps = rng.standard_normal((len(xb), model.latent))
            loss, rec, kl, cache = model.forward(xb, eps)
            opt.step(model.p, model.backward(cache))
            tot += np.array([loss, rec, kl]) * len(xb)
        history.append(tot / n)
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss {history[-1][0]:8.3f}  recon {history[-1][1]:8.3f}  kl {history[-1][2]:6.3f}")
    return np.array(history)
