"""PHOENIX EXPERIMENTAL HARNESS — self-contained, validated.

Everything the experiment queue needs, in one file. Drop it in the repo root (it imports
only `phoenix.soa` and numpy) and validate it before spending compute:

    python phoenix_harness.py

That runs `validate()`, which reproduces a known result with a known-exact value. If it
does not reproduce, the harness is wrong and nothing measured with it means anything.

CONVENTIONS (kept identical everywhere so numbers are comparable across experiments):
  - dt = 1 ms, tau = 3 ms, refractory 2 ms, delays uniform 1-8 ms -> MAX DELAY SPAN 8 ms
  - background drive: per cell, probability 0.005/tick = 5.0 Hz, weight 30 mV,
    suprathreshold. ANY measured rate must be compared against 5 Hz, NOT zero.
  - readout: per-cell time of FIRST spike in [lag, lag+window); non-firing cells get
    `window` as a sentinel
  - decoder: leave-one-out nearest centroid, vectorized (chance = 1/K)
  - saturation ceiling is 333 Hz (refractory limit)
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, ".")
from phoenix.soa import SoANetwork

DRIVE_HZ = 5.0
MAX_DELAY = 8.0
SAT_HZ = 333.0


# --------------------------------------------------------------------------- decoder

def loo_nearest_centroid_pred(X, y, block=512):
    """||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2; ||x||^2 is constant per row so it is
    dropped. Leave-one-out corrects only the sample's own centroid.
    Verified identical to the naive implementation; ~70 s at K=1280 / n=7680 / d=64000.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    classes, idx = np.unique(y, return_inverse=True)
    k = len(classes)
    sums = np.zeros((k, X.shape[1]))
    np.add.at(sums, idx, X)
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    cents = sums / counts[:, None]
    cn = np.einsum("ij,ij->i", cents, cents)
    pred = np.empty(len(X), dtype=classes.dtype)
    for lo in range(0, len(X), block):
        hi = min(lo + block, len(X))
        Xb = X[lo:hi]
        d = cn[None, :] - 2.0 * (Xb @ cents.T)
        for r in range(hi - lo):
            c = idx[lo + r]
            if counts[c] > 1:
                own = (sums[c] - Xb[r]) / (counts[c] - 1)
                d[r, c] = own @ own - 2.0 * (Xb[r] @ own)
            else:
                d[r, c] = np.inf
        pred[lo:hi] = classes[np.argmin(d, axis=1)]
    return pred


def accuracy(X, y):
    return float(np.mean(loo_nearest_centroid_pred(X, y) == np.asarray(y)))


def mutual_information_bits(y_true, y_pred, K):
    """WARNING: severely biased upward at low sample counts. At K=80 with 480 samples,
    SHUFFLED labels score 3.59 bits when the true value is 0. Always subtract
    `shuffle_floor` or use accuracy instead."""
    joint = np.zeros((K, K))
    for t, p in zip(y_true, y_pred):
        joint[t, p] += 1
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))


def shuffle_floor(n_samples, K, reps=5, seed=0):
    """Bias floor of the MI estimator: MI of random labels, whose true value is 0."""
    r = np.random.RandomState(seed)
    return float(np.mean([mutual_information_bits(r.randint(0, K, n_samples),
                                                  r.randint(0, K, n_samples), K)
                          for _ in range(reps)]))


# ----------------------------------------------------------------------- construction

def build(n, fan_out=20, g_exc=6.98, log_sd=2.0, g_ratio=4.0, f_inh=0.2,
          tau=3.0, seed=0, mode="rec"):
    """mode='rec'  : every cell projects (recurrent)
       mode='ff'   : only the given source cells project - a genuine one-hop delay line
       mode='none' : no synapses at all ('no network', NOT 'feedforward')

    log_sd is the spread of the log-normal weight draw. mu = ln(mean) - sd^2/2 pins the
    mean, so gain and spread move independently. log_sd~0.3 approximates the original
    uniform(0.5,1.5); the operating band opens from ~1.0 upward.
    """
    rng = np.random.RandomState(seed)
    net = SoANetwork(dt=1.0)
    for i in range(n):
        net.add_cell(i, tau=tau, refractory_period=2.0)
    is_inh = rng.random_sample(n) < f_inh
    if mode == "none" or fan_out == 0:
        net._ensure_built()
        return net, is_inh
    src = np.arange(n)
    pre = np.repeat(src, fan_out)
    post = rng.randint(0, n, len(src) * fan_out)
    mu = np.log(g_exc) - log_sd ** 2 / 2.0
    w = rng.lognormal(mu, log_sd, len(pre))
    d = rng.uniform(1.0, MAX_DELAY, len(pre))
    w[is_inh[pre]] *= -g_ratio
    net.add_synapses_bulk(pre, post, w, d, 1.0, 20.0)
    net._ensure_built()
    return net, is_inh


def build_feedforward(n, sources, fan_out=20, g_exc=6.98, log_sd=2.0, seed=0):
    """One-hop delay line: only `sources` project, and ONLY to non-sources.

    Restricting targets is not cosmetic. If a target can itself be a source, it
    re-projects and you get two hops - memory to 2*MAX_DELAY, and the control silently
    stops being feedforward. With heavy-tailed weights that second hop fires almost
    every time: the first version of this function scored 82.5% at lag 12 ms where the
    structural answer is exactly chance.

    So this has memory up to MAX_DELAY by construction and provably none past it - the
    fair control for memory-span claims. (fan_out=0 is 'no network', which trivially has
    no memory and inflates the apparent value of recurrence.)
    """
    rng = np.random.RandomState(seed)
    net = SoANetwork(dt=1.0)
    for i in range(n):
        net.add_cell(i, tau=3.0, refractory_period=2.0)
    src = np.unique(np.asarray(sources))
    targets = np.setdiff1d(np.arange(n), src)   # one hop only: never back into a source
    if len(targets) == 0:
        raise ValueError("no non-source cells left to project onto")
    pre = np.repeat(src, fan_out)
    post = targets[rng.randint(0, len(targets), len(pre))]
    mu = np.log(g_exc) - log_sd ** 2 / 2.0
    w = rng.lognormal(mu, log_sd, len(pre))
    d = rng.uniform(1.0, MAX_DELAY, len(pre))
    net.add_synapses_bulk(pre, post, w, d, 1.0, 20.0)
    net._ensure_built()
    return net


def reset(net):
    c = net.cells
    c.Vm = c.Vrest.copy()
    c.refractory_until[:] = 0.0
    c.last_spike_time[:] = -np.inf
    c.t = 0.0
    net._heap.clear()
    net._external[:] = 0.0


# ------------------------------------------------------------------------ measurement

def firing_rate(net, n, ticks=6000, p_drive=0.005, drive_w=30.0, seed=1234, n_bins=10):
    """Returns (steady-state Hz over the second half, per-bin Hz).

    ALWAYS inspect the bins. A configuration held ~6 Hz for 3000 ticks and then ignited
    to 117 Hz; a mean over the whole run hides that completely.
    """
    reset(net)
    rng = np.random.RandomState(seed)
    per = np.zeros(ticks)
    for t in range(ticks):
        for i in np.flatnonzero(rng.random_sample(n) < p_drive):
            net.inject(int(i), drive_w)
        per[t] = len(net.step())
    bins = per.reshape(n_bins, ticks // n_bins).sum(1) / (n * (ticks // n_bins) * 1e-3)
    tail = per[ticks // 2:]
    return tail.sum() / (n * len(tail) * 1e-3), bins


def _fingerprints(net, n, cues, lags, trials, window, p_drive, drive_w, cue_w,
                  jitter, n_ticks, seed):
    rng = np.random.RandomState(seed)
    X = {L: [] for L in lags}
    y = []
    for ci, cue in enumerate(cues):
        for _ in range(trials):
            reset(net)
            onset = 5 + (rng.randint(-jitter, jitter + 1) if jitter else 0)
            first = {L: np.full(n, np.nan) for L in lags}
            for tick in range(n_ticks):
                for i in np.flatnonzero(rng.random_sample(n) < p_drive):
                    net.inject(int(i), drive_w)
                if tick == onset:
                    for c in cue:
                        net.inject(int(c), cue_w)
                fired = net.step()
                if not fired:
                    continue
                fi = np.fromiter((net._id2idx[i] for i in fired), dtype=np.int64,
                                 count=len(fired))
                for L in lags:
                    lo = onset + L
                    if lo <= tick < lo + window:
                        f = first[L]
                        f[fi[np.isnan(f[fi])]] = tick - lo
            for L in lags:
                X[L].append(np.where(np.isnan(first[L]), float(window), first[L]))
            y.append(ci)
    return X, np.array(y)


def memory_task(net, n, cues, lags=(8, 12, 20), trials=10, window=10, p_drive=0.005,
                drive_w=30.0, cue_w=45.0, jitter=2, n_ticks=60, seed=99, readout=None):
    """Discriminate WHICH cue was presented, read out at [lag, lag+window).
    Chance = 1/len(cues)."""
    X, y = _fingerprints(net, n, cues, list(lags), trials, window, p_drive, drive_w,
                         cue_w, jitter, n_ticks, seed)
    out = {}
    for L in lags:
        M = np.stack(X[L])
        if readout is not None and readout < n:
            M = M[:, np.random.RandomState(7).choice(n, readout, replace=False)]
        out[L] = accuracy(M, y)
    return out


def capacity(net, n, K=80, cue_size=None, trials=6, lag=20, window=10, p_drive=0.005,
             drive_w=30.0, cue_w=45.0, jitter=2, n_ticks=50, seed=0, readout=None):
    """Capacity task: K distinct cues, single readout lag. Returns (accuracy, chance).

    Defaults follow the established protocol: cue = 4% of cells, readout = 25% of N,
    lag 20 ms (past the 8 ms delay span, so this measures what the network HOLDS rather
    than what it relays). Scale K with N or the task saturates near N~89,000 and you
    measure the ceiling instead of the network.
    """
    if cue_size is None:
        cue_size = max(1, n // 25)
    if readout is None:
        readout = max(1, n // 4)
    rng = np.random.RandomState(seed)
    cues = [rng.choice(n, cue_size, replace=False) for _ in range(K)]
    X, y = _fingerprints(net, n, cues, [lag], trials, window, p_drive, drive_w, cue_w,
                         jitter, n_ticks, seed + 1)
    M = np.stack(X[lag])
    if readout < n:
        M = M[:, np.random.RandomState(7).choice(n, readout, replace=False)]
    return accuracy(M, y), 1.0 / K


# --------------------------------------------------------------------------- validate

def label_shuffled_floor(X, y, reps=5, seed=0):
    """Empirical no-information accuracy for THIS decoder and design.

    Do not assume the floor is 1/K. Leave-one-out nearest centroid sits BELOW chance
    when there is no signal: removing a sample from its own centroid pushes that
    centroid away from it, so it is systematically less likely to be assigned to its own
    class. With 5 classes and 8 trials each the floor is ~8%, not 20%.

    (This is also the explanation for a previously reported "20.0% +/- 0.0" at chance
    25% - that number WAS this artifact, not a coincidence.)
    """
    r = np.random.RandomState(seed)
    out = []
    for _ in range(reps):
        yp = y.copy()
        r.shuffle(yp)
        out.append(accuracy(X, yp))
    return float(np.mean(out)), float(np.std(out))


def validate(n=1000, seeds=(0, 1, 2), verbose=True):
    """Reproduce a structurally known result before trusting anything else.

    A one-hop feedforward delay line has memory up to MAX_DELAY (8 ms) and is
    STRUCTURALLY incapable past it. So:
      - within the span it must be near-perfect
      - past the span it must be indistinguishable from the label-shuffled floor

    If past-span accuracy sits ABOVE that floor, the harness is leaking the stimulus
    (readout window overlapping the direct response, wrong onset, incomplete reset, or a
    "feedforward" control that quietly allows a second hop) and every number it produces
    is meaningless.
    """
    K = 5
    res = {4: [], 12: [], 20: []}
    floors = {12: [], 20: []}
    for sd in seeds:
        rng = np.random.RandomState(100 + sd)
        cues = [rng.choice(n, 40, replace=False) for _ in range(K)]
        net = build_feedforward(n, np.concatenate(cues), seed=sd)
        X, y = _fingerprints(net, n, cues, [4, 12, 20], 8, 10, 0.005, 30.0, 45.0,
                             2, 60, 99 + sd)
        for L in res:
            res[L].append(accuracy(np.stack(X[L]), y))
        for L in floors:
            floors[L].append(label_shuffled_floor(np.stack(X[L]), y, seed=sd)[0])

    within_ok = np.mean(res[4]) > 0.80
    past_ok = all(abs(np.mean(res[L]) - np.mean(floors[L])) < 0.08 for L in (12, 20))
    ok = within_ok and past_ok

    if verbose:
        print(f"  {K} cues, nominal chance {1/K:.0%}, max delay span {MAX_DELAY:.0f} ms")
        print(f"  lag=  4 ms  {np.mean(res[4]):6.1%}  (within span -> expect near-perfect)"
              f"  {'OK' if within_ok else 'FAIL'}")
        for L in (12, 20):
            m, f = np.mean(res[L]), np.mean(floors[L])
            good = abs(m - f) < 0.08
            print(f"  lag={L:>3} ms  {m:6.1%}  vs shuffled floor {f:6.1%}"
                  f"   (past span -> must match floor)  {'OK' if good else 'FAIL'}")
        print(f"  -> harness {'VALID' if ok else 'INVALID - do not trust any measurement'}")
    return ok


if __name__ == "__main__":
    print("=== HARNESS VALIDATION ===")
    sys.exit(0 if validate() else 1)
