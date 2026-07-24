"""CUE SPARSITY — the never-swept variable, and a direct test of the sparse-
distributed-representation principle.

Cue size has been fixed at 4% of cells since the beginning and NEVER swept. Sparse
distributed codes get their capacity FROM sparsity: fewer active cells per pattern ->
less pairwise overlap between patterns -> more patterns separable before interference.
At N=64k a 4% cue is 2,560 active cells with ~102 cells of expected pairwise overlap
between two random cues; 1% is 640 cells (~6.4 overlap), 0.5% is 320 (~1.6). If the
sparse-distributed principle holds HERE (it has been asserted, never measured), sparser
cues should push the capacity break (K_max) OUT.

DESIGN: hold N and the config fixed; pick K AT or JUST PAST the measured break (where
accuracy is most sensitive); sweep cue fraction in {4%, 2%, 1%, 0.5%}. Report accuracy
vs cue fraction with the shuffled floor at each point. A rise as cues get sparser =
sparsity buys capacity. Flat/declining = the principle does not operate in this regime
(also a real result). Everything else identical to the capacity runs (lognormal config,
readout 25%, lag 20, LOO nearest-centroid, measured floor).

This is FAR cheaper than a 256k point and tests a first-principles claim directly.

Usage:  python experiments/task_cue_sparsity.py N K [cue_frac ...] [--seeds s ...]
        (default cue_fracs: 0.04 0.02 0.01 0.005)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from phoenix_harness import (  # noqa: E402
    accuracy, build, label_shuffled_floor, reset, _fingerprints,
)

RESULTS = os.path.join("experiments", "task_cue_sparsity_results.json")
CONFIG = dict(fan_out=20, g_exc=6.98, log_sd=2.0, mode="rec")
TRIALS = 6
LAG = 20
WINDOW = 10
N_TICKS = 50
P_DRIVE = 0.005
DRIVE_W = 30.0
CUE_W = 45.0
JITTER = 2


def measure_rate(net, n, ticks=1500, seed=1234):
    reset(net)
    rng = np.random.RandomState(seed)
    fired = 0
    for _ in range(ticks):
        for i in np.flatnonzero(rng.random_sample(n) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        fired += len(net.step())
    return fired / (n * ticks * 1e-3)


def run_one(n, K, cue_frac, seed):
    cue_size = max(1, int(round(n * cue_frac)))
    readout = max(1, n // 4)
    t0 = time.perf_counter()
    net, _ = build(n, seed=seed, **CONFIG)
    rate = measure_rate(net, n)
    rng = np.random.RandomState(seed)
    cues = [rng.choice(n, cue_size, replace=False) for _ in range(K)]
    X, y = _fingerprints(net, n, cues, [LAG], TRIALS, WINDOW, P_DRIVE, DRIVE_W,
                         CUE_W, JITTER, N_TICKS, seed + 1)
    M = np.stack(X[LAG])
    if readout < n:
        M = M[:, np.random.RandomState(7).choice(n, readout, replace=False)]
    acc = accuracy(M, y)
    floor, floor_sd = label_shuffled_floor(M, y, reps=3, seed=seed)
    # expected pairwise overlap between two random cues of this size (hypergeometric mean)
    exp_overlap = cue_size * cue_size / n
    return dict(N=n, K=K, cue_frac=cue_frac, cue_size=cue_size, seed=seed,
                accuracy=acc, floor=floor, floor_sd=floor_sd, chance=1.0 / K,
                rate_hz=rate, exp_pairwise_overlap=exp_overlap,
                seconds=time.perf_counter() - t0)


def main():
    n = int(sys.argv[1]); K = int(sys.argv[2])
    rest = sys.argv[3:]
    if "--seeds" in rest:
        i = rest.index("--seeds")
        fracs = [float(x) for x in rest[:i]] or [0.04, 0.02, 0.01, 0.005]
        seeds = [int(x) for x in rest[i + 1:]]
    else:
        fracs = [float(x) for x in rest] or [0.04, 0.02, 0.01, 0.005]
        seeds = [0]

    existing = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            existing = json.load(f)

    print(f"Cue-sparsity sweep at N={n:,}, K={K} (chance {1/K:.3%}); "
          f"cue_fracs={fracs}; seeds={seeds}\n", flush=True)
    for seed in seeds:
        for frac in fracs:
            done = any(r["N"] == n and r["K"] == K and r["seed"] == seed
                       and abs(r["cue_frac"] - frac) < 1e-9 for r in existing)
            if done:
                print(f"  skip cue={frac:.1%} seed={seed} (done)", flush=True)
                continue
            r = run_one(n, K, frac, seed)
            existing.append(r)
            with open(RESULTS, "w") as f:
                json.dump(existing, f, indent=1)
            print(f"  cue={frac:6.2%} ({r['cue_size']:>5} cells, ~{r['exp_pairwise_overlap']:.1f} "
                  f"overlap)  seed={seed}  acc={r['accuracy']:7.2%}  floor={r['floor']:6.2%}  "
                  f"rate={r['rate_hz']:.1f}Hz  ({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
