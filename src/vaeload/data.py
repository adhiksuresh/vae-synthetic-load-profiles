"""Synthetic household daily load profiles.

The real smart-meter data used in the original project cannot be shared, so
this module generates a stand-in dataset with the features that matter for
the experiment:

* 48 half-hourly readings per day (kWh per half hour)
* a morning peak and a larger evening peak whose timing and size vary by household
* weekday/weekend differences (later, flatter mornings at weekends)
* seasonal heating/cooling load
* multiplicative noise and occasional short spikes (kettle, oven, EV)

Every household has its own persistent "personality" (base load, peak sizes,
peak times), so profiles are correlated within a household, as in real data.
"""

import numpy as np

STEPS_PER_DAY = 48
HOURS = np.arange(STEPS_PER_DAY) / 2.0  # 0.0, 0.5, ..., 23.5


def _bump(centre, width, height):
    """Smooth peak (Gaussian bump) on the half-hour grid, wrapping round midnight."""
    d = np.minimum(np.abs(HOURS - centre), 24 - np.abs(HOURS - centre))
    return height * np.exp(-0.5 * (d / width) ** 2)


def generate_profiles(n_households: int = 200, n_days: int = 60, seed: int = 0):
    """Return (profiles, meta).

    profiles : array (n_households * n_days, 48), kWh per half hour, all > 0
    meta     : dict of arrays with household id, day index and weekend flag
    """
    rng = np.random.default_rng(seed)
    rows, hh_ids, day_ids, weekend = [], [], [], []

    for h in range(n_households):
        base = rng.lognormal(np.log(0.12), 0.35)          # standby load
        morning_t = rng.normal(7.5, 0.7)                   # morning peak time
        evening_t = rng.normal(19.5, 1.0)                  # evening peak time
        morning_h = rng.lognormal(np.log(0.25), 0.4)
        evening_h = rng.lognormal(np.log(0.55), 0.4)
        heat_sens = rng.uniform(0.0, 0.25)                 # sensitivity to cold days

        for d in range(n_days):
            is_weekend = (d % 7) in (5, 6)
            season = np.cos(2 * np.pi * d / 365.0)          # 1 = mid-winter
            mt = morning_t + (1.5 if is_weekend else 0.0) + rng.normal(0, 0.3)
            et = evening_t + rng.normal(0, 0.4)
            mh = morning_h * (0.7 if is_weekend else 1.0) * rng.lognormal(0, 0.2)
            eh = evening_h * rng.lognormal(0, 0.2)
            midday = 0.08 * base / 0.12 * (1.6 if is_weekend else 1.0)

            p = (
                base
                + _bump(mt, 1.0, mh)
                + _bump(et, 1.6, eh)
                + _bump(13.0, 2.5, midday)
                + heat_sens * max(season, 0) * (_bump(7.0, 2.0, 1.0) + _bump(20.0, 3.0, 1.0))
            )
            p = p * rng.lognormal(0, 0.12, STEPS_PER_DAY)    # measurement / behaviour noise
            n_spikes = rng.poisson(1.2)
            for _ in range(n_spikes):
                p[rng.integers(12, STEPS_PER_DAY)] += rng.uniform(0.2, 1.0)

            rows.append(p)
            hh_ids.append(h)
            day_ids.append(d)
            weekend.append(is_weekend)

    meta = {
        "household": np.array(hh_ids),
        "day": np.array(day_ids),
        "weekend": np.array(weekend),
    }
    return np.array(rows), meta


def split_by_household(profiles, meta, val_fraction=0.2, test_fraction=0.2, seed=0):
    """Train/validation/test split by household.

    Splitting by household (not by day) means the model is always evaluated
    on households it has never seen, which is the realistic use case.
    """
    rng = np.random.default_rng(seed)
    households = rng.permutation(np.unique(meta["household"]))
    n_test = int(len(households) * test_fraction)
    n_val = int(len(households) * val_fraction)
    test_hh, val_hh = households[:n_test], households[n_test:n_test + n_val]
    is_test = np.isin(meta["household"], test_hh)
    is_val = np.isin(meta["household"], val_hh)
    is_train = ~(is_test | is_val)
    return profiles[is_train], profiles[is_val], profiles[is_test]
