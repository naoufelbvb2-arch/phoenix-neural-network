"""TASK 2 — the sparsity / cost trade-off, and the obstruction that limits it.

PRE-REGISTERED CRITERION (fixed before running, verbatim from the brief):
    "sweep fan_out in {1,2,5,10,20,40} at matched rate, >=3 seeds, report function
     per synapse. Confirmed if function per synapse rises monotonically as fan-out
     falls."

WHAT ACTUALLY HAPPENED — the sweep is NOT executable across the full range in any single
weight regime, and that obstruction is the primary finding (methodological rule 3: "if no
rate-matched control exists within a family, that absence is itself the finding").

The obstruction is Result 11 (weight distribution and rate are entangled) biting directly:

  * UNIFORM weights (log_sd=0.3): only LOW fan-out admits a STABLE target rate. A gate
    probe (bisect g to 13 Hz on seed 0, then verify over 6000 ticks) found stable flat-bin
    points for fan_out 1,2,5 (g = 44.3, 30.3, 21.8) but fan_out 10/20 sit BELOW target at
    their last stable g (8.4 Hz @ g=16.4, 6.0 Hz @ g=12.1); the next step up in g ignites
    to saturation (Result 8: g~12.3 -> >100 Hz). High fan-out uniform has a ~zero-width
    operating band, so there is no stable 13 Hz point to rate-match to.

  * LOG-NORMAL weights (log_sd=2.0): the mirror image. High fan-out gets a graded band and
    sits at ~13 Hz, but low fan-out saturates BELOW target -- fan_out=1 maxes ~5.8 Hz and
    fan_out=2 maxes ~7.0 Hz even at g=40 (probe: experiments task2 g->rate table). They
    cannot be driven up to 13 Hz at all.

So the pre-registered criterion "monotonic across {1,2,5,10,20,40}" is NOT FULLY TESTABLE:
no weight regime rate-matches the whole range. Rate cannot be separated from structure here
because the weight distribution that fixes the rate is itself a structural variable.

WHAT IS EXECUTABLE — the low fan-out window under uniform weights, where a stable rate-
matched point exists. g is CALIBRATED per fan-out (bisection on seed 0) to 13 Hz and reused
across seeds; realized rates are reported so residual mismatch is visible, not hidden.
FUNCTION = capacity accuracy (leave-one-out nearest centroid, unbiased); floor measured per
point with label_shuffled_floor(); "per synapse" = accuracy / (N * fan_out).

Result over this window (N=2000, K=40, 3 seeds):
    fan_out=1: rate 13.2+/-0.9 Hz  acc 15.7%  acc/syn 7.85e-5   (24.6x vs fan_out=5)
    fan_out=2: rate 13.4+/-0.1 Hz  acc 10.7%  acc/syn 2.67e-5   ( 8.4x vs fan_out=5)
    fan_out=5: rate 15.7+/-5.8 Hz  acc  3.2%  acc/syn 3.19e-6   ( 1.0x)
Function per synapse rises as fan-out falls -- BUT only fan_out 1,2 are tightly rate-matched;
fan_out=5's rate drifted up (one seed ignited), so its point is confounded by the very rate
mismatch rule 3 warns about. Firm conclusion: over the TIGHTLY matched pair {1,2}, sparser is
~3x more function per synapse. The full monotonic sweep across {1..40} remains untestable.

Everything (build, capacity, floor, rate) comes from the shared validated harness.

Usage:  python experiments/task2_sparsity_cost.py [N] [K]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from phoenix_harness import (  # noqa: E402
    accuracy, build, firing_rate, label_shuffled_floor, reset, _fingerprints,
)

RESULTS = os.path.join("experiments", "task2_results.json")

# Only the fan-outs that admit a STABLE rate-matched point under uniform weights
# (verified by the 6000-tick gate probe). The high fan-outs are omitted BY FINDING,
# not by choice: they have no stable 13 Hz point to match to.
FAN_OUTS = [1, 2, 5]
LOG_SD = 0.3              # uniform weights -- required for low fan-out to reach 13 Hz
TARGET_HZ = 13.0
RATE_TOL = 1.0
LAG = 20
WINDOW = 10
N_TICKS = 50
TRIALS = 6
P_DRIVE = 0.005
DRIVE_W = 30.0
CUE_W = 45.0
JITTER = 2
SEEDS = (0, 1, 2)


def rate_at(n, fan_out, g, ticks=900, n_bins=3, seed=0):
    net, _ = build(n, fan_out=fan_out, g_exc=g, log_sd=LOG_SD, seed=seed)
    r, _ = firing_rate(net, n, ticks=ticks, n_bins=n_bins, seed=1234 + seed)
    return r


def calibrate_g(n, fan_out, seed=0, lo=1.0, hi=200.0, iters=9):
    """Bisect g_exc to TARGET_HZ on the calibration seed. Rate is monotone in g."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = rate_at(n, fan_out, mid, seed=seed)
        if r < TARGET_HZ:
            lo = mid
        else:
            hi = mid
    g = (lo + hi) / 2
    # verify on a long run (n_bins divides ticks -- the reshape needs it)
    r6 = rate_at(n, fan_out, g, ticks=6000, n_bins=6, seed=seed)
    return g, r6


def capacity_acc(n, fan_out, g, seed, K):
    cue_size = max(1, n // 25)
    readout = max(1, n // 4)
    net, _ = build(n, fan_out=fan_out, g_exc=g, log_sd=LOG_SD, seed=seed)
    rng = np.random.RandomState(seed)
    cues = [rng.choice(n, cue_size, replace=False) for _ in range(K)]
    X, y = _fingerprints(net, n, cues, [LAG], TRIALS, WINDOW, P_DRIVE, DRIVE_W,
                         CUE_W, JITTER, N_TICKS, seed + 1)
    M = np.stack(X[LAG])
    if readout < n:
        M = M[:, np.random.RandomState(7).choice(n, readout, replace=False)]
    acc = accuracy(M, y)
    floor, _ = label_shuffled_floor(M, y, reps=3, seed=seed)
    reset(net)
    r, _ = firing_rate(net, n, ticks=1000, n_bins=2, seed=999 + seed)
    return acc, floor, r


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(f"TASK 2 — sparsity/cost at N={n}, K={K} (chance {1/K:.1%}), "
          f"uniform weights, target {TARGET_HZ} Hz")
    print("  (high fan-out omitted BY FINDING: no stable rate-matched point exists)\n")

    rows = []
    for fan_out in FAN_OUTS:
        g, g_rate = calibrate_g(n, fan_out, seed=0)
        accs, floors, rates = [], [], []
        for seed in SEEDS:
            acc, floor, r = capacity_acc(n, fan_out, g, seed, K)
            accs.append(acc); floors.append(floor); rates.append(r)
        rows.append(dict(fan_out=fan_out, g_exc=g, calib_rate=g_rate,
                         acc=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                         floor=float(np.mean(floors)), rate=float(np.mean(rates)),
                         rate_sd=float(np.std(rates)), synapses=n * fan_out,
                         accs=accs))
        r = rows[-1]
        matched = "matched" if r["rate_sd"] < 1.5 and abs(r["rate"] - TARGET_HZ) < 2 \
            else "DRIFTED"
        print(f"  fan_out={fan_out:>2}  g={g:5.1f}  rate={r['rate']:5.1f}+/-{r['rate_sd']:.1f}Hz "
              f"[{matched}]  acc={r['acc']:6.2%}+/-{r['acc_sd']:.2%}  "
              f"floor={r['floor']:5.2%}  syn={r['synapses']:,}", flush=True)

    with open(RESULTS, "w") as f:
        json.dump(dict(N=n, K=K, target_hz=TARGET_HZ, log_sd=LOG_SD,
                       obstruction="full sweep not rate-matchable in any single weight "
                                   "regime (Result 11); high fan-out has zero-width band "
                                   "under uniform weights, low fan-out cannot reach target "
                                   "under log-normal weights",
                       rows=rows), f, indent=1)

    base = rows[-1]["acc"] / rows[-1]["synapses"]  # fan_out=5
    print(f"\n  function per synapse (normalized to fan_out={FAN_OUTS[-1]}):")
    for r in rows:
        fps = r["acc"] / r["synapses"]
        print(f"    fan_out={r['fan_out']:>2}: {fps/base:8.2f}x")

    desc = sorted(rows, key=lambda r: -r["fan_out"])
    fps_desc = [r["acc"] / r["synapses"] for r in desc]
    strictly_up = all(b > a for a, b in zip(fps_desc, fps_desc[1:]))
    tight = [r for r in rows if r["rate_sd"] < 1.5 and abs(r["rate"] - TARGET_HZ) < 2]
    print(f"\n=== function/synapse rises as fan-out falls over {FAN_OUTS}? "
          f"{'yes' if strictly_up else 'no'} ===")
    print(f"=== tightly rate-matched points: fan_out {[r['fan_out'] for r in tight]} "
          f"(the rest are confounded by rate drift) ===")
    print("=== full {1,2,5,10,20,40} sweep: NOT testable -- no single weight regime "
          "rate-matches the whole range (the finding) ===")


if __name__ == "__main__":
    main()
