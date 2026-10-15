"""Leakage-safe splits: hash of (run, subrun) -> train/val/test; nested fraction subsets."""
import hashlib

import numpy as np

SPLIT_SEED = "qlfm-v1"


def _u01(run, subrun):
    h = hashlib.sha256(f"{SPLIT_SEED}:{run}:{subrun}".encode()).digest()
    return int.from_bytes(h[:8], "little") / 2 ** 64


def make_splits(run, subrun, frac_val=0.05, frac_test=0.05):
    """Return dict of global index arrays; all events of a (run, subrun) share a partition."""
    keys = np.stack([run, subrun], 1)
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    u = np.array([_u01(r, s) for r, s in uk])[inv.ravel()]
    test = u < frac_test
    val = (u >= frac_test) & (u < frac_test + frac_val)
    idx = np.arange(len(run))
    return {"train": idx[~(test | val)], "val": idx[val], "test": idx[test]}


def nested_subset(indices, frac_of_total, n_total, seed=0):
    """First round(frac_of_total * n_total) events of a fixed permutation of `indices`.

    Nested: a smaller fraction is always a subset of a larger one."""
    perm = np.random.default_rng(seed).permutation(indices)
    k = min(len(perm), int(round(frac_of_total * n_total)))
    return np.sort(perm[:k])
