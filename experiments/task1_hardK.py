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

PRE-REGISTERED BREAK CRITERION (fixed 2026-07-24, BEFORE any sweep row landed, so the
break is not chosen after seeing the curve):
    K_max(N) = the largest K at which lognormal accuracy >= 50%, obtained by LINEAR
    INTERPOLATION between the two bracketing sweep points (the K just above and just
    below the 0.50 crossing). The measured shuffled floor is reported at EVERY K.
    The full accuracy-vs-K curve is reported, not just K_max — the shape is the result.

PRE-REGISTERED SCALING TEST (the actual question — does capacity grow with N?):
    Run the IDENTICAL sweep at N=16,000 and N=64,000. Compare K_max(64k) vs K_max(16k):
      * K_max(64k) ~= 4 x K_max(16k)  -> capacity scales with N (linear).
      * K_max flat across the 4x size step -> capacity does NOT scale with N.
    This is confound-free: K_max is an absolute pattern count, so no chance-denominator
    (1/K) enters the comparison at all. A 256k point, if run later, is a THIRD point
    confirming the slope — never the sole evidence.

Run 16k first (cheap: 16k rows were the fast ones), then 64k. N=256,000 points are ~a
day each and are placed at the K the 16k/64k slope flags, not sprayed blindly.

Usage:  python experiments/task1_hardK.py N [K ...] [--seeds s ...]
        python experiments/task1_hardK.py --analyze         # compute K_max per (N,seed)
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


def kmax_interpolate(ks, accs, thresh=0.50):
    """Pre-registered K_max: largest K with accuracy >= thresh, linear-interpolated
    between the bracketing sweep points. Returns (kmax, note)."""
    pts = sorted(zip(ks, accs))
    ks = [k for k, _ in pts]; accs = [a for _, a in pts]
    if accs[0] < thresh:
        return None, f"already below {thresh:.0%} at smallest K={ks[0]} (acc {accs[0]:.1%}) — K_max < {ks[0]}"
    if accs[-1] >= thresh:
        return None, f"still >= {thresh:.0%} at largest K={ks[-1]} (acc {accs[-1]:.1%}) — K_max > {ks[-1]} (extend sweep)"
    for i in range(len(ks) - 1):
        if accs[i] >= thresh > accs[i + 1]:
            # linear interpolation of the crossing in K
            frac = (accs[i] - thresh) / (accs[i] - accs[i + 1])
            return ks[i] + frac * (ks[i + 1] - ks[i]), f"crosses between K={ks[i]} ({accs[i]:.1%}) and K={ks[i+1]} ({accs[i+1]:.1%})"
    return None, "no crossing found"


def analyze():
    if not os.path.exists(RESULTS):
        print("no results yet"); return
    with open(RESULTS) as f:
        rows = json.load(f)
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        if r["condition"] == "lognormal":
            by[(r["N"], r["seed"])].append((r["K"], r["accuracy"], r["floor"], r["rate_hz"]))
    print("PRE-REGISTERED K_max = largest K with lognormal accuracy >= 50% (interpolated)\n")
    kmax_by_N = defaultdict(list)
    for (N, seed) in sorted(by):
        pts = sorted(by[(N, seed)])
        print(f"  N={N:,} seed={seed}:")
        for K, acc, fl, rate in pts:
            print(f"    K={K:>6} (={K/(N//200):5.1f}x N/200)  acc={acc:7.2%}  floor={fl:6.2%}  rate={rate:.1f}Hz")
        km, note = kmax_interpolate([p[0] for p in pts], [p[1] for p in pts])
        print(f"    -> K_max = {km if km is None else round(km,1)}  ({note})\n")
        if km is not None:
            kmax_by_N[N].append(km)
    if len(kmax_by_N) >= 2:
        Ns = sorted(kmax_by_N)
        print("  SCALING:")
        for N in Ns:
            print(f"    K_max({N:,}) = {np.mean(kmax_by_N[N]):.0f}")
        lo, hi = Ns[0], Ns[-1]
        ratio = np.mean(kmax_by_N[hi]) / np.mean(kmax_by_N[lo])
        nratio = hi / lo
        print(f"    K_max ratio {hi//1000}k/{lo//1000}k = {ratio:.2f}x  (N ratio {nratio:.0f}x) "
              f"-> {'scales with N' if ratio > 0.7*nratio else 'does NOT scale linearly with N'}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze(); return
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
