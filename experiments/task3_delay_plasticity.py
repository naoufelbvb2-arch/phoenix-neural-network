"""TASK 3 — delay plasticity. See PREREG_task3_delay_plasticity.md (locked before running).

Two post-spike-free rules (A arrival-consensus, B use-dependent speedup), four arms each
(cue-trained / frozen-random / shuffled / random-trained), N=2000, K=160, lag=12 (ceiling-checked
51.7% baseline), >=5 seeds. Criterion: confirmed only if cue-trained beats BOTH random-trained AND
shuffled. Diagnostics reported pre+post: delay distribution AND per-post-cell arrival dispersion.
Rate-matched against the POST-training rate.

Delays live in net._syn_delay (CSR order); set_delays() rewrites them and rebuilds the bucket ring.
Arrivals are RECONSTRUCTED from the recorded spike train x the synapse table (no post-spike is read
by the learning rule — the whole design point).

Usage:  python experiments/task3_delay_plasticity.py <A|B> [--seeds s ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from phoenix.soa import _ragged_ranges  # noqa: E402
from phoenix_harness import (  # noqa: E402
    accuracy, build, label_shuffled_floor, reset, _fingerprints,
)

N = 2000
K = 160
LAG = 12
WINDOW = 10
READOUT = N // 4
CUE = N // 25
D_MIN, D_MAX = 1.0, 8.0
ONSET = 5
W_RESP = 20            # arrivals in [onset, onset+W_resp] drive the updates (evoked volley)
W_COIN = 3.0           # coincidence window (Rule B)
N_TICKS_TRAIN = 30
EPOCHS = 15
ETA_A = 0.15           # consensus step (proportional)
ETA_B = 0.20           # speedup step (per-epoch, usage-weighted)
ETA_RELAX = 0.05
P_DRIVE, DRIVE_W, CUE_W, JITTER = 0.005, 30.0, 45.0, 2
RESULTS = os.path.join("experiments", "task3_results.json")


# ----------------------------------------------------------------- net delay plumbing
def synapse_arrays(net):
    """(pre_idx per synapse, post_idx, delay) in CSR order."""
    indptr = net._syn_indptr
    pre = np.repeat(np.arange(N), np.diff(indptr))
    return pre, net._syn_post_idx.copy(), net._syn_delay.copy()


def set_delays(net, new_delay):
    """Rewrite per-synapse delays and rebuild the bucket ring (bucket = max(1, ceil(delay)))."""
    net._syn_delay = np.clip(new_delay, D_MIN, D_MAX)
    off = np.maximum(1, np.ceil(net._syn_delay).astype(np.int64))
    net._syn_bucket_off = off
    net._max_off = int(off.max()) if off.size else 1
    net._ring_size = net._max_off + 1
    net._ring = [[] for _ in range(net._ring_size)]


# ----------------------------------------------------------------- presentation + arrivals
def present(net, cue, seed):
    """One presentation; return the recorded spike train (idx array, time array)."""
    reset(net)
    rng = np.random.RandomState(seed)
    onset = ONSET + rng.randint(-JITTER, JITTER + 1)
    sp_idx, sp_t = [], []
    for tick in range(N_TICKS_TRAIN):
        for i in np.flatnonzero(rng.random_sample(N) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        if tick == onset and cue is not None:
            for i in cue:
                net.inject(int(i), CUE_W)
        fired = net.step()          # neuron_ids == idx for this build
        if fired:
            sp_idx.extend(fired); sp_t.extend([net.cells.t] * len(fired))
    return np.asarray(sp_idx, dtype=np.int64), np.asarray(sp_t, dtype=np.float64), onset


def arrivals(net, sp_idx, sp_t, onset):
    """Reconstruct (synapse_index, post_idx, arrival_time) in the response window, from the
    spike train x the synapse table. Reads NO postsynaptic spike — arrivals are pre-driven."""
    if sp_idx.size == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0)
    indptr = net._syn_indptr
    counts = indptr[sp_idx + 1] - indptr[sp_idx]
    nz = counts > 0
    if not nz.any():
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0)
    sp_idx, sp_t, counts = sp_idx[nz], sp_t[nz], counts[nz]
    syn = _ragged_ranges(indptr[sp_idx], counts)
    a_time = np.repeat(sp_t, counts) + net._syn_delay[syn]
    a_post = net._syn_post_idx[syn]
    m = (a_time >= onset) & (a_time <= onset + W_RESP)
    return syn[m], a_post[m], a_time[m]


# ----------------------------------------------------------------- the two rules
def sumcount_ruleA(net, syn, a_post, a_time):
    """Arrival-consensus contributions for ONE presentation: per synapse, the summed nudge
    (median_of_post - arrival) and the count. train() averages sum/count over the epoch so the
    step is the MEAN nudge (bounded), not the sum (which slams into d_max). Median over post
    cells with >=2 arrivals; post-spike is never read."""
    E = net._syn_delay.size
    s = np.zeros(E); c = np.zeros(E)
    if syn.size == 0:
        return s, c
    counts = np.bincount(a_post, minlength=N)
    order = np.lexsort((a_time, a_post))               # group by post, then time
    sp, st = a_post[order], a_time[order]
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    med_idx = starts + counts // 2                     # middle element per group (median-ish)
    has2 = counts >= 2
    median = np.zeros(N)
    median[has2] = st[med_idx[has2]]
    contrib = has2[a_post]                             # only post cells with >=2 arrivals
    np.add.at(s, syn[contrib], median[a_post[contrib]] - a_time[contrib])
    np.add.at(c, syn[contrib], 1.0)
    return s, c


def usage_ruleB(net, syn, a_post, a_time):
    """Per-presentation: which synapses contributed to a coincidence (post had >=2 arrivals within
    W_COIN of each other). Returns a boolean over all synapses."""
    used = np.zeros(net._syn_delay.size, dtype=bool)
    if syn.size < 2:
        return used
    # Sort by (post, time); an arrival is coincident if its neighbour in the SAME post group is
    # within W_COIN. Fully vectorized (no per-arrival Python loop).
    order = np.lexsort((a_time, a_post))
    sp, st, so = a_post[order], a_time[order], syn[order]
    same_next = sp[1:] == sp[:-1]
    near_next = same_next & ((st[1:] - st[:-1]) <= W_COIN)   # element i is near i+1
    coincident = np.zeros(order.size, dtype=bool)
    coincident[:-1] |= near_next                              # near the following arrival
    coincident[1:] |= near_next                              # ... makes the following one coincident too
    used[so[coincident]] = True
    return used


# ----------------------------------------------------------------- train / eval / diagnose
def train(net, patterns, rule, seed):
    d_init = net._syn_delay.copy()
    E = d_init.size
    for epoch in range(EPOCHS):
        if rule == "A":
            S = np.zeros(E); C = np.zeros(E)
            for p, cue in enumerate(patterns):
                si, stime, onset = present(net, cue, seed * 100000 + epoch * 1000 + p)
                syn, a_post, a_time = arrivals(net, si, stime, onset)
                s, c = sumcount_ruleA(net, syn, a_post, a_time)
                S += s; C += c
            delta = ETA_A * np.where(C > 0, S / np.maximum(C, 1), 0.0)   # MEAN nudge per synapse
            set_delays(net, net._syn_delay + delta)
        else:  # B
            usage = np.zeros(E)
            for p, cue in enumerate(patterns):
                si, stime, onset = present(net, cue, seed * 100000 + epoch * 1000 + p)
                syn, a_post, a_time = arrivals(net, si, stime, onset)
                usage += usage_ruleB(net, syn, a_post, a_time)
            frac = usage / len(patterns)
            delta = -ETA_B * frac + ETA_RELAX * (1 - frac) * (d_init - net._syn_delay)
            set_delays(net, net._syn_delay + delta)
    return net


def arrival_dispersion(net, cues, seed):
    """Mean over post cells of the sd of input arrival times within the response window — the
    quantity Rule A acts on. Reported pre+post (bound-fraction cannot see its collapse)."""
    disp = []
    for c in cues[:40]:                                 # a sample of cues is enough
        si, stime, onset = present(net, c, seed + 55)
        syn, a_post, a_time = arrivals(net, si, stime, onset)
        if a_post.size == 0:
            continue
        counts = np.bincount(a_post, minlength=N)
        for j in np.flatnonzero(counts >= 2):
            disp.append(a_time[a_post == j].std())
    return float(np.mean(disp)) if disp else 0.0


def diag(net, cues, seed):
    d = net._syn_delay
    return dict(delay_mean=float(d.mean()), delay_sd=float(d.std()),
                frac_dmin=float(np.mean(d <= D_MIN + 1e-9)),
                frac_dmax=float(np.mean(d >= D_MAX - 1e-9)),
                arr_dispersion=arrival_dispersion(net, cues, seed))


def measure_rate(net, seed):
    reset(net)
    rng = np.random.RandomState(seed)
    fired = 0
    for _ in range(1000):
        for i in np.flatnonzero(rng.random_sample(N) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        fired += len(net.step())
    return fired / (N * 1000 * 1e-3)


def evaluate(net, cues, seed):
    X, y = _fingerprints(net, N, cues, [LAG], 6, WINDOW, P_DRIVE, DRIVE_W, CUE_W, JITTER, 50, seed + 1)
    M = np.stack(X[LAG])[:, np.random.RandomState(7).choice(N, READOUT, replace=False)]
    acc = accuracy(M, y)
    floor, _ = label_shuffled_floor(M, y, reps=3, seed=seed)
    return acc, floor


def calibrate_g(target_rate, seed, lo=3.0, hi=20.0):
    for _ in range(7):
        mid = (lo + hi) / 2
        net, _ = build(N, g_exc=mid, seed=seed)
        r = measure_rate(net, seed + 3)
        if r < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_rule(rule, seed):
    rng = np.random.RandomState(seed)
    cues = [rng.choice(N, CUE, replace=False) for _ in range(K)]
    rand_inputs = [rng.choice(N, CUE, replace=False) for _ in range(K)]  # random "cues" for arm 4

    out = {}
    # cue-trained
    net, _ = build(N, seed=seed)
    pre_diag = diag(net, cues, seed)
    train(net, cues, rule, seed)
    post_rate = measure_rate(net, seed + 7)
    post_diag = diag(net, cues, seed)
    acc_cue, fl_cue = evaluate(net, cues, seed)
    trained_delays = net._syn_delay.copy()
    out["cue_trained"] = dict(acc=acc_cue, floor=fl_cue, rate=post_rate,
                              pre=pre_diag, post=post_diag)

    # frozen-random, rate-matched to the cue-trained POST-training rate
    g_match = calibrate_g(post_rate, seed)
    net_fr, _ = build(N, g_exc=g_match, seed=seed)
    acc_fr, fl_fr = evaluate(net_fr, cues, seed)
    out["frozen_random"] = dict(acc=acc_fr, floor=fl_fr, rate=measure_rate(net_fr, seed + 7),
                                g_exc=g_match, diag=diag(net_fr, cues, seed))

    # shuffled: cue-trained magnitudes, delays permuted across synapses
    net_sh, _ = build(N, seed=seed)
    set_delays(net_sh, np.random.RandomState(seed + 1).permutation(trained_delays))
    acc_sh, fl_sh = evaluate(net_sh, cues, seed)
    out["shuffled"] = dict(acc=acc_sh, floor=fl_sh, diag=diag(net_sh, cues, seed))

    # random-trained: plasticity ON, trained on RANDOM input, evaluated on cues
    net_rt, _ = build(N, seed=seed)
    train(net_rt, rand_inputs, rule, seed + 999)
    acc_rt, fl_rt = evaluate(net_rt, cues, seed)
    out["random_trained"] = dict(acc=acc_rt, floor=fl_rt, rate=measure_rate(net_rt, seed + 7),
                                 diag=diag(net_rt, cues, seed))
    return out


def main():
    rule = sys.argv[1]
    rest = sys.argv[2:]
    seeds = [int(x) for x in rest[rest.index("--seeds") + 1:]] if "--seeds" in rest else [0]
    existing = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    for seed in seeds:
        if any(r["rule"] == rule and r["seed"] == seed for r in existing):
            print(f"  skip rule {rule} seed {seed}", flush=True); continue
        t0 = time.perf_counter()
        res = run_rule(rule, seed)
        res.update(rule=rule, seed=seed, seconds=time.perf_counter() - t0)
        existing.append(res)
        json.dump(existing, open(RESULTS, "w"), indent=1)
        c, r, s, f = (res["cue_trained"]["acc"], res["random_trained"]["acc"],
                      res["shuffled"]["acc"], res["frozen_random"]["acc"])
        print(f"  RULE {rule} seed {seed}: cue={c:.1%} random-trained={r:.1%} shuffled={s:.1%} "
              f"frozen={f:.1%}  disp {res['cue_trained']['pre']['arr_dispersion']:.2f}->"
              f"{res['cue_trained']['post']['arr_dispersion']:.2f}ms "
              f"dmin={res['cue_trained']['post']['frac_dmin']:.0%} "
              f"dmax={res['cue_trained']['post']['frac_dmax']:.0%} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
