"""PARTITION TEST — does GLOBAL CONNECTIVITY add capacity, or is a big net four small ones?

The decisive test Control A could not settle. Control A held the readout SIZE fixed, but that also
made it a 4x smaller FRACTION of the 64k net, so it cannot separate "the network stores no more"
from "a fixed readout cannot extract it". Here the readout scales with N (16000 = 25%), exactly as
a downstream area would, and the ONLY thing that differs between the two 64k systems is whether
synapses cross block boundaries:

  connected   : one 64k network.                 cue 2560 (4%),  readout 16000 (25%).
  partitioned : four DISCONNECTED 16k blocks.     cue 640/blk (2560 total), readout 4000/blk (16000).

Matched: cell count, synapse count (fan_out=20 each way, targets just stay in-block), weight and
delay distributions, E/I, cue set, readout set. Only global connectivity differs.

  partitioned ~= connected  -> the 64k net is four 16k nets in a trenchcoat; integration adds
                               nothing; scaling is pointless even with a scaled readout. Closes Qn A.
  connected  >  partitioned -> integration across scale is real capacity — the first measured
                               argument this project has produced FOR a large network.

Both nets share the SAME cue cells and readout cells (same seeds), so the comparison is clean.

Usage:  python experiments/task1_partition.py <connected|partitioned> K [K ...] [--seeds s ...]
        python experiments/task1_partition.py --analyze
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from phoenix.soa import SoANetwork  # noqa: E402
from phoenix_harness import (  # noqa: E402
    accuracy, firing_rate, label_shuffled_floor, reset, _fingerprints,
)

N = 64000
N_BLOCKS = 4
FAN_OUT = 20
CUE = 2560          # 4% of N (640 per block)
READOUT = 16000     # 25% of N (4000 per block)
LAG, WINDOW, N_TICKS, TRIALS = 20, 10, 50, 6
P_DRIVE, DRIVE_W, CUE_W, JITTER = 0.005, 30.0, 45.0, 2


def build(mode, seed, g_exc=6.98, log_sd=2.0, g_ratio=4.0, f_inh=0.2, tau=3.0):
    """One 64k net. mode='connected' -> global targets; 'partitioned' -> in-block targets.
    Weights/delays/E-I are the SAME draws either way; only the post indices differ."""
    rng = np.random.RandomState(seed)
    is_inh = rng.random_sample(N) < f_inh
    pre = np.repeat(np.arange(N), FAN_OUT)
    mu = np.log(g_exc) - log_sd ** 2 / 2.0
    w = rng.lognormal(mu, log_sd, len(pre))
    d = rng.uniform(1.0, 8.0, len(pre))
    w[is_inh[pre]] *= -g_ratio
    blk = N // N_BLOCKS
    if mode == "connected":
        post = rng.randint(0, N, len(pre))
    elif mode == "partitioned":
        post = (pre // blk) * blk + rng.randint(0, blk, len(pre))  # stay inside the block
    else:
        raise ValueError(mode)
    net = SoANetwork(dt=1.0)
    for i in range(N):
        net.add_cell(i, tau=tau, refractory_period=2.0)
    net.add_synapses_bulk(pre, post, w, d, 1.0, 20.0)
    net._ensure_built()
    return net


def run_one(mode, K, seed):
    t0 = time.perf_counter()
    net = build(mode, seed)
    rng = np.random.RandomState(seed)               # SAME cues for both modes (seed-shared)
    cues = [rng.choice(N, CUE, replace=False) for _ in range(K)]
    X, y = _fingerprints(net, N, cues, [LAG], TRIALS, WINDOW, P_DRIVE, DRIVE_W,
                         CUE_W, JITTER, N_TICKS, seed + 1)
    M = np.stack(X[LAG])
    M = M[:, np.random.RandomState(7).choice(N, READOUT, replace=False)]  # SAME readout cells
    acc = accuracy(M, y)
    floor, _ = label_shuffled_floor(M, y, reps=3, seed=seed)
    reset(net)
    rate, _ = firing_rate(net, N, ticks=1000, n_bins=2, seed=999 + seed)
    return dict(mode=mode, N=N, K=K, seed=seed, accuracy=acc, floor=floor,
                chance=1.0 / K, rate_hz=rate, seconds=time.perf_counter() - t0)


def kmax_interp(ks, accs, thresh=0.50):
    pts = sorted(zip(ks, accs)); ks = [k for k, _ in pts]; accs = [a for _, a in pts]
    if accs[0] < thresh:
        return None, f"<{ks[0]} (already {accs[0]:.1%} at K={ks[0]})"
    if accs[-1] >= thresh:
        return None, f">{ks[-1]} (still {accs[-1]:.1%} at K={ks[-1]})"
    for i in range(len(ks) - 1):
        if accs[i] >= thresh > accs[i + 1]:
            frac = (accs[i] - thresh) / (accs[i] - accs[i + 1])
            return ks[i] + frac * (ks[i + 1] - ks[i]), f"[{ks[i]},{ks[i+1]}]"
    return None, "?"


def results_path(mode):
    return os.path.join("experiments", f"partition_{mode}.json")


def analyze():
    from collections import defaultdict
    by = defaultdict(list)
    for mode in ("connected", "partitioned"):
        p = results_path(mode)
        if os.path.exists(p):
            for r in json.load(open(p)):
                by[r["mode"]].append((r["K"], r["accuracy"], r["floor"]))
    print("PARTITION TEST — K_max (>=50%, interpolated)\n")
    kmax = {}
    for mode in ("connected", "partitioned"):
        pts = sorted(by.get(mode, []))
        if not pts:
            print(f"  {mode}: (no rows)\n"); continue
        print(f"  {mode}:")
        for K, acc, fl in pts:
            print(f"    K={K:>5}  acc={acc:7.2%}  floor={fl:6.2%}")
        km, note = kmax_interp([p[0] for p in pts], [p[1] for p in pts])
        kmax[mode] = km
        print(f"    -> K_max = {km if km is None else round(km,1)} {note}\n")
    if kmax.get("connected") and kmax.get("partitioned"):
        r = kmax["connected"] / kmax["partitioned"]
        verdict = ("connected > partitioned: integration ADDS capacity" if r > 1.3 else
                   "connected ~= partitioned: big net is four small ones (integration adds nothing)")
        print(f"  K_max(connected)/K_max(partitioned) = {r:.2f} -> {verdict}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze(); return
    mode = sys.argv[1]
    rest = sys.argv[2:]
    if "--seeds" in rest:
        i = rest.index("--seeds")
        Ks = [int(x) for x in rest[:i]]; seeds = [int(x) for x in rest[i + 1:]]
    else:
        Ks = [int(x) for x in rest]; seeds = [0]
    if not Ks:
        Ks = [80, 320, 1280]
    path = results_path(mode)
    existing = json.load(open(path)) if os.path.exists(path) else []
    print(f"PARTITION {mode}: N={N:,}, {N_BLOCKS} blocks, cue={CUE}, readout={READOUT}; "
          f"Ks={Ks}; seeds={seeds}\n", flush=True)
    for seed in seeds:
        for K in Ks:
            if any(r["mode"] == mode and r["K"] == K and r["seed"] == seed for r in existing):
                print(f"  skip {mode} K={K} seed={seed} (done)", flush=True); continue
            r = run_one(mode, K, seed)
            existing.append(r)
            json.dump(existing, open(path, "w"), indent=1)
            print(f"  {mode:<11} K={K:>5} seed={seed}  acc={r['accuracy']:7.2%} "
                  f"floor={r['floor']:6.2%} rate={r['rate_hz']:.1f}Hz ({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
