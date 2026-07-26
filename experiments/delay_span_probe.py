"""DELAY-SPAN PROBE — is the delay lever held multi-hop memory or feedforward relay reach?
See PREREG_delay_span_probe.md (locked). This module: core memory-depth measurement at fixed N —
recurrent net vs graded feedforward chains (k=1..4), max over k at each lag (non-tautological),
depth unit = lag/mean_delay, window ~ 2.2*mean_delay, all rate-matched to a common rate.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from phoenix.soa import SoANetwork
from phoenix_harness import accuracy, label_shuffled_floor, reset

TAU, REFRAC = 3.0, 2.0
FAN_OUT, LOG_SD, G_RATIO, F_INH = 20, 2.0, 4.0, 0.2
P_DRIVE, DRIVE_W, CUE_W, JITTER = 0.005, 30.0, 45.0, 2


def _wire(n, pre, post, w, d, tau=TAU):
    net = SoANetwork(dt=1.0)
    for i in range(n):
        net.add_cell(i, tau=tau, refractory_period=REFRAC)
    net.add_synapses_bulk(pre, post, w, d, 1.0, 20.0)
    net._ensure_built()
    return net


def build_recurrent(n, seed, dlo, dhi, g_exc):
    rng = np.random.RandomState(seed)
    is_inh = rng.random_sample(n) < F_INH
    pre = np.repeat(np.arange(n), FAN_OUT)
    post = rng.randint(0, n, len(pre))
    mu = np.log(g_exc) - LOG_SD ** 2 / 2.0
    w = rng.lognormal(mu, LOG_SD, len(pre)); d = rng.uniform(dlo, dhi, len(pre)); w[is_inh[pre]] *= -G_RATIO
    return _wire(n, pre, post, w, d), np.arange(n)          # input pool = all cells (random cue)


def build_chain(n, k, seed, dlo, dhi, g_exc):
    rng = np.random.RandomState(seed); L = n // (k + 1)
    is_inh = rng.random_sample(n) < F_INH
    pre_all, post_all = [], []
    for j in range(k):
        src = np.arange(j * L, (j + 1) * L)
        pre_all.append(np.repeat(src, FAN_OUT))
        post_all.append(rng.randint((j + 1) * L, (j + 2) * L, L * FAN_OUT))
    pre = np.concatenate(pre_all); post = np.concatenate(post_all)
    mu = np.log(g_exc) - LOG_SD ** 2 / 2.0
    w = rng.lognormal(mu, LOG_SD, len(pre)); d = rng.uniform(dlo, dhi, len(pre)); w[is_inh[pre]] *= -G_RATIO
    return _wire(n, pre, post, w, d), np.arange(L)          # input pool = layer 0


def rate_of(net, n, seed, ticks=800):
    reset(net); rng = np.random.RandomState(seed); fired = 0
    for _ in range(ticks):
        for i in np.flatnonzero(rng.random_sample(n) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        fired += len(net.step())
    return fired / (n * ticks * 1e-3)


def calibrate(builder, n, seed, dlo, dhi, target=12.0, lo=2.0, hi=40.0):
    for _ in range(7):
        g = (lo + hi) / 2
        net = builder(n, seed, dlo, dhi, g)[0]
        if rate_of(net, n, seed + 1) < target: lo = g
        else: hi = g
    return (lo + hi) / 2


def fingerprints(net, n, input_pool, cue_size, K, lag, window, n_ticks, seed):
    rng = np.random.RandomState(seed)
    cues = [rng.choice(input_pool, cue_size, replace=False) for _ in range(K)]
    X, y = [], []
    trials = 6
    for ci, cue in enumerate(cues):
        for _ in range(trials):
            reset(net)
            onset = 5 + rng.randint(-JITTER, JITTER + 1)
            first = np.full(n, float(window))
            seen = np.zeros(n, bool)
            for tick in range(n_ticks):
                for i in np.flatnonzero(rng.random_sample(n) < P_DRIVE):
                    net.inject(int(i), DRIVE_W)
                if tick == onset:
                    for i in cue: net.inject(int(i), CUE_W)
                for f in net.step():
                    lo = onset + lag
                    if lo <= net.cells.t < lo + window and not seen[f]:
                        first[f] = net.cells.t - lo; seen[f] = True
            X.append(first); y.append(ci)
    return np.array(X), np.array(y)


def acc_at(net, n, input_pool, cue_size, readout_idx, K, lag, window, n_ticks, seed):
    X, y = fingerprints(net, n, input_pool, cue_size, K, lag, window, n_ticks, seed)
    M = X[:, readout_idx]
    return accuracy(M, y)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seeds = [0, 1, 2]
    dlo, dhi = 5.0, 8.0                 # generator: uniform[5,8], mean 6.5
    mean_delay = (dlo + dhi) / 2
    window = int(round(2.2 * mean_delay))
    K, cue_size = 120, 80
    depths = [1.0, 1.5, 2.0, 2.5, 3.0]  # lag in MEAN-HOPS
    lags = [int(round(d * mean_delay)) for d in depths]
    print(f"DELAY-SPAN PROBE core: N={n}, gen=uniform[{dlo},{dhi}] mean={mean_delay}, window={window}, "
          f"K={K}, cue={cue_size}; lags(mean-hops)={list(zip(depths,lags))}\n", flush=True)

    rows = []
    for seed in seeds:
        readout = np.arange(n)                                  # full net (fair to chains' active layer)
        g_rec = calibrate(build_recurrent, n, seed, dlo, dhi)
        rec = build_recurrent(n, seed, dlo, dhi, g_rec)[0]
        chains = {}
        for k in (1, 2, 3, 4):
            g = calibrate(lambda *a: build_chain(a[0], k, a[1], a[2], a[3], a[4]), n, seed, dlo, dhi)
            chains[k] = build_chain(n, k, seed, dlo, dhi, g)
        nt = 5 + max(lags) + window + 5
        for depth, lag in zip(depths, lags):
            a_rec = acc_at(rec, n, np.arange(n), cue_size, readout, K, lag, window, nt, seed + 10)
            a_ch = {}
            for k, (cnet, pool) in chains.items():
                a_ch[k] = acc_at(cnet, n, pool, cue_size, readout, K, lag, window, nt, seed + 10)
            best_k = max(a_ch, key=a_ch.get)
            rows.append(dict(seed=seed, depth=depth, lag=lag, recurrent=a_rec,
                             best_chain=a_ch[best_k], best_k=best_k, chains=a_ch))
            print(f"  seed {seed} depth={depth} (lag {lag}): recurrent={a_rec:5.1%}  "
                  f"best-chain={a_ch[best_k]:5.1%} (k={best_k})  "
                  f"[{' '.join(f'{k}:{v:.0%}' for k,v in a_ch.items())}]", flush=True)
    json.dump(rows, open(os.path.join("experiments", f"delay_span_core_N{n}.json"), "w"), indent=1)
    # summary
    import collections
    by = collections.defaultdict(list)
    for r in rows: by[r["depth"]].append((r["recurrent"], r["best_chain"]))
    print("\n  MEAN over seeds — recurrent vs best-chain:")
    for depth in depths:
        rc = np.mean([x[0] for x in by[depth]]); bc = np.mean([x[1] for x in by[depth]])
        verdict = "recurrent HOLDS more" if rc > bc + 0.05 else "== feedforward"
        print(f"    depth={depth}: recurrent={rc:5.1%}  best-chain={bc:5.1%}  -> {verdict}")


if __name__ == "__main__":
    main()
