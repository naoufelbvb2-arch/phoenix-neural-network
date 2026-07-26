"""TASK 3 addendum — is Rule A's gain a DELAY DISTRIBUTION, generable a priori (no plasticity)?

Rule A took the task 51.7% (uniform-random delays) -> 68% (cue-trained). Not cue memory
(cue ~= random-trained), but a local rule found a better delay CONFIGURATION than uniform.
This is the log-normal pattern a third time (synaptic scaling: the useful weight spread was
better drawn a priori than produced by the controller). So: characterize the converged delays,
then DRAW delays from that distribution into a fresh net with NO training and re-run the task.

  direct-gen ~= cue-trained  -> uniform 1-8ms is a poor default; the useful delay distribution is
                               X and it needs no plasticity (second positive of the log-normal class).
  direct-gen <  cue-trained  -> per-synapse STRUCTURE matters beyond the marginal (but the shuffle
                               control already says most of the gain is the marginal).

Rate is matched by construction: direct-gen uses the converged marginal -> same mean delay -> same
rate as cue-trained. Reported anyway.

Usage:  python experiments/task3_delay_distribution.py [--seeds s ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import importlib.util
_spec = importlib.util.spec_from_file_location("t3", os.path.join("experiments", "task3_delay_plasticity.py"))
t3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(t3)
from phoenix_harness import build  # noqa: E402

RESULTS = os.path.join("experiments", "task3_distribution_results.json")


def characterize(net, d):
    post_fanin = np.bincount(net._syn_post_idx, minlength=t3.N)      # in-degree per post cell
    syn_fanin = post_fanin[net._syn_post_idx].astype(float)          # per synapse
    corr_fanin = float(np.corrcoef(d, syn_fanin)[0, 1])
    pct = np.percentile(d, [10, 25, 50, 75, 90])
    return dict(mean=float(d.mean()), sd=float(d.std()),
                p10=float(pct[0]), p25=float(pct[1]), p50=float(pct[2]),
                p75=float(pct[3]), p90=float(pct[4]),
                frac_dmin=float(np.mean(d <= t3.D_MIN + 1e-9)),
                frac_dmax=float(np.mean(d >= t3.D_MAX - 1e-9)),
                corr_delay_postfanin=corr_fanin)


def run(seed):
    rng = np.random.RandomState(seed)
    cues = [rng.choice(t3.N, t3.CUE, replace=False) for _ in range(t3.K)]

    # 1) cue-trained Rule A -> converged delays (reproducible from seed)
    net, _ = build(t3.N, seed=seed)
    d_uniform = net._syn_delay.copy()
    t3.train(net, cues, "A", seed)
    d_conv = net._syn_delay.copy()
    acc_cue = t3.evaluate(net, cues, seed)[0]
    rate_cue = t3.measure_rate(net, seed + 7)
    E = d_conv.size

    # 2) direct-generation: fresh net, delays RESAMPLED iid from the converged marginal, NO training
    net_gen, _ = build(t3.N, seed=seed)
    d_gen = np.random.RandomState(seed + 321).choice(d_conv, size=E, replace=True)
    t3.set_delays(net_gen, d_gen)
    acc_gen = t3.evaluate(net_gen, cues, seed)[0]
    rate_gen = t3.measure_rate(net_gen, seed + 7)

    # 3) uniform baseline (untrained, standard) — the poor default, for reference
    net_u, _ = build(t3.N, seed=seed)
    acc_uniform = t3.evaluate(net_u, cues, seed)[0]

    return dict(seed=seed, acc_cue_trained=acc_cue, acc_direct_gen=acc_gen,
                acc_uniform=acc_uniform, rate_cue=rate_cue, rate_gen=rate_gen,
                dist_uniform=characterize(net, d_uniform),
                dist_converged=characterize(net, d_conv))


def main():
    rest = sys.argv[1:]
    seeds = [int(x) for x in rest[rest.index("--seeds") + 1:]] if "--seeds" in rest else [0, 1, 2, 3, 4]
    existing = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    for seed in seeds:
        if any(r["seed"] == seed for r in existing):
            print(f"  skip seed {seed}", flush=True); continue
        t0 = time.perf_counter(); r = run(seed); r["seconds"] = time.perf_counter() - t0
        existing.append(r); json.dump(existing, open(RESULTS, "w"), indent=1)
        dc = r["dist_converged"]
        print(f"  seed {seed}: uniform={r['acc_uniform']:.1%}  direct-gen={r['acc_direct_gen']:.1%}  "
              f"cue-trained={r['acc_cue_trained']:.1%}  |  converged delays mean={dc['mean']:.2f} "
              f"p50={dc['p50']:.1f} dmax={dc['frac_dmax']:.0%} corr(delay,postfanin)={dc['corr_delay_postfanin']:+.2f} "
              f"({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
