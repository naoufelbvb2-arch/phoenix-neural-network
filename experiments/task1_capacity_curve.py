"""TASK 1 — does capacity keep growing with N? (decision-critical)

PRE-REGISTERED CRITERION (fixed before running, verbatim from the brief):
    "measure at N = 16,000 / 64,000 / 256,000, >=3 seeds each. Growth confirmed if
     accuracy-relative-to-chance rises across all three; refuted if it plateaus or
     declines at the top. Report whichever you get."

Config from result 12: log-normal log_sd=2.0, g_exc=6.98, 20% inhibitory at g_ratio=4,
cue = 4% of cells, readout = 25% of N, lag 20 ms, window 10 ms. K scales with N (K=N/200)
so the task does not saturate near N~89,000 and measure the ceiling instead of the net.

MANDATORY CONTROLS at every N:
  fan_out=0        — must be ~floor. If it is not, the readout is leaking the stimulus
                     and the whole measurement is invalid.
  sparse-uniform   — fan_out=2, g_exc~30, near-uniform weights. Confirms or refutes that
                     the log-normal advantage stays a constant ~x2 rather than widening.

Accuracy only (unbiased under LOO); MI is not used — it is severely biased at these
sample counts. The no-information floor is MEASURED per condition with
label_shuffled_floor(), never assumed to be 1/K.

Results append to experiments/task1_results.json so sizes can be accumulated across
invocations (the run is long; see the budget note in the report).

Usage:  python experiments/task1_capacity_curve.py N [seed ...]
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

RESULTS = os.path.join("experiments", "task1_results.json")
TRIALS = 6
LAG = 20
WINDOW = 10
N_TICKS = 50
P_DRIVE = 0.005
DRIVE_W = 30.0
CUE_W = 45.0
JITTER = 2

CONDITIONS = {
    "lognormal":      dict(fan_out=20, g_exc=6.98, log_sd=2.0, mode="rec"),
    "sparse_uniform": dict(fan_out=2,  g_exc=30.0, log_sd=0.3, mode="rec"),
    "fan_out_0":      dict(fan_out=0,  g_exc=6.98, log_sd=2.0, mode="none"),
}


def measure_rate(net, n, ticks=1500, seed=1234):
    """Short steady-state rate. Compare against the 5 Hz drive baseline, not zero."""
    reset(net)
    rng = np.random.RandomState(seed)
    fired = 0
    for _ in range(ticks):
        for i in np.flatnonzero(rng.random_sample(n) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        fired += len(net.step())
    return fired / (n * ticks * 1e-3)


def run_one(n, condition, seed):
    kwargs = CONDITIONS[condition]
    K = max(2, n // 200)
    cue_size = max(1, n // 25)
    readout = max(1, n // 4)

    t0 = time.perf_counter()
    net, _ = build(n, seed=seed, **kwargs)
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
    elapsed = time.perf_counter() - t0
    return dict(N=n, condition=condition, seed=seed, K=K, accuracy=acc,
                floor=floor, floor_sd=floor_sd, chance=1.0 / K, rate_hz=rate,
                readout=readout, cue_size=cue_size, seconds=elapsed)


def main():
    n = int(sys.argv[1])
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]

    existing = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            existing = json.load(f)

    for seed in seeds:
        for condition in CONDITIONS:
            done = any(r["N"] == n and r["condition"] == condition and r["seed"] == seed
                       for r in existing)
            if done:
                print(f"  skip N={n} {condition} seed={seed} (already done)", flush=True)
                continue
            r = run_one(n, condition, seed)
            existing.append(r)
            with open(RESULTS, "w") as f:
                json.dump(existing, f, indent=1)
            print(f"  N={r['N']:>7,} {r['condition']:<15} seed={seed} K={r['K']:>4} "
                  f"acc={r['accuracy']:7.2%} floor={r['floor']:6.2%} "
                  f"chance={r['chance']:6.2%} rate={r['rate_hz']:7.1f}Hz "
                  f"({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
