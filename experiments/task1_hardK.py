"""TASK 1 (addendum) — where does capacity actually BREAK? A K-sweep at fixed N.

WHY: the N/200 capacity curve is becoming uninformative at the top. accuracy is
within ~9 points of ceiling at N=64k (90.7%) and the accuracy/chance RATIO
(40x -> 290x) is increasingly driven by the SHRINKING CHANCE DENOMINATOR (K grows
with N), not by the network improving — the same inflation trap as the MI bias.
"Still rising" cannot be claimed from a ratio whose denominator is collapsing.

So hold N FIXED and push K WELL ABOVE N/200 until accuracy falls toward the floor.
That break point is the real capacity, independent of the chance denominator:
  * if accuracy stays high until K reaches some large fraction of N and then drops,
    the high N/200 accuracy reflects genuine capacity;
  * if accuracy is already sliding at K just past N/200, the ratio was inflated and
    the honest claim is only "holds >= N/200 patterns at >= 90%", NOT "still rising".

DESIGN: same lognormal config and readout/cue/lag/decoder as the capacity curve
(so numbers are comparable), only K varies. fan_out_0 control at each K confirms the
floor. Separate results file (task1_hardK_results.json) so this never collides with
the pre-registered curve run. Resumable; K-sweep dedups on (N, condition, seed, K).

Run at N=64,000 first (feasible, ~1-2 h/point). N=256,000 points are ~a day each and
should be placed at the K the 64k sweep flags as the break, not sprayed blindly.

Usage:  python experiments/task1_hardK.py N [K ...] [--seeds s ...]
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

RESULTS = os.path.join("experiments", "task1_hardK_results.json")
TRIALS = 6
LAG = 20
WINDOW = 10
N_TICKS = 50
P_DRIVE = 0.005
DRIVE_W = 30.0
CUE_W = 45.0
JITTER = 2

CONDITIONS = {
    "lognormal": dict(fan_out=20, g_exc=6.98, log_sd=2.0, mode="rec"),
    "fan_out_0": dict(fan_out=0,  g_exc=6.98, log_sd=2.0, mode="none"),
}


def measure_rate(net, n, ticks=1500, seed=1234):
    reset(net)
    rng = np.random.RandomState(seed)
    fired = 0
    for _ in range(ticks):
        for i in np.flatnonzero(rng.random_sample(n) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        fired += len(net.step())
    return fired / (n * ticks * 1e-3)


def run_one(n, condition, seed, K):
    kwargs = CONDITIONS[condition]
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
                readout=readout, cue_size=cue_size, seconds=elapsed,
                K_over_Nover200=K / max(1, n // 200))


def main():
    n = int(sys.argv[1])
    rest = sys.argv[2:]
    if "--seeds" in rest:
        i = rest.index("--seeds")
        Ks = [int(x) for x in rest[:i]]
        seeds = [int(x) for x in rest[i + 1:]]
    else:
        Ks = [int(x) for x in rest]
        seeds = [0]
    if not Ks:
        Ks = [n // 200, 4 * (n // 200), 16 * (n // 200), 64 * (n // 200)]

    existing = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            existing = json.load(f)

    print(f"K-sweep at N={n:,} (N/200={n//200}); Ks={Ks}; seeds={seeds}\n", flush=True)
    for seed in seeds:
        for K in Ks:
            for condition in CONDITIONS:
                done = any(r["N"] == n and r["condition"] == condition
                           and r["seed"] == seed and r["K"] == K for r in existing)
                if done:
                    print(f"  skip N={n} {condition} seed={seed} K={K} (done)", flush=True)
                    continue
                r = run_one(n, condition, seed, K)
                existing.append(r)
                with open(RESULTS, "w") as f:
                    json.dump(existing, f, indent=1)
                print(f"  N={n:>7,} {condition:<10} seed={seed} K={K:>6} "
                      f"(={r['K_over_Nover200']:.0f}x N/200)  acc={r['accuracy']:7.2%} "
                      f"floor={r['floor']:6.2%} chance={r['chance']:.3%} "
                      f"rate={r['rate_hz']:.1f}Hz ({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
