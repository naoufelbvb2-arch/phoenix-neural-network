"""CONSTRUCTIVE TEST — build the architecture the analysis implies, run it on the benchmarks.

The negative results say the recurrent machinery is inessential. The constructive claim: an explicit
FEEDFORWARD delay-line + coincidence-detector net — NO cycles, NO inhibition, NO plasticity/homeostasis,
log-normal weights, matched mean delay — matches or beats the recurrent architecture on the repo's
benchmarks. This module: K_max capacity (the decisive one), feedforward (k-layer chain, f_inh=0) vs
recurrent (with inhibition), N=2000, readout at ~2 mean-hops.

Both rate-matched to a common rate. K_max = largest K with accuracy >= 50% (interpolated).
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, ".")
import importlib.util
_spec = importlib.util.spec_from_file_location("P", os.path.join("experiments", "delay_span_probe.py"))
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

N = 2000
DLO, DHI = 5.0, 8.0
MEAN = (DLO + DHI) / 2
LAG = 13                       # ~2 mean-hops
WINDOW = int(round(2.2 * MEAN))
K_LIST = [320, 640, 1280, 2560]
CUE = 80
READOUT = 300                  # BOUNDED (avoid the readout-dimensionality inflation), architecture-fair
SEEDS = [0, 1, 2]


def kmax(ks, accs, thr=0.50):
    pts = sorted(zip(ks, accs)); ks = [k for k, _ in pts]; a = [x for _, x in pts]
    if a[0] < thr: return None, f"<{ks[0]}"
    if a[-1] >= thr: return None, f">{ks[-1]}"
    for i in range(len(ks) - 1):
        if a[i] >= thr > a[i + 1]:
            f = (a[i] - thr) / (a[i] - a[i + 1]); return ks[i] + f * (ks[i + 1] - ks[i]), f"[{ks[i]},{ks[i+1]}]"
    return None, "?"


def run_arch(arch, seed):
    nt = 5 + LAG + WINDOW + 5
    rr = np.random.RandomState(seed + 5)
    if arch == "recurrent":
        P.F_INH = 0.2                                        # standard recurrent, WITH inhibition
        g = P.calibrate(P.build_recurrent, N, seed, DLO, DHI)
        net, pool = P.build_recurrent(N, seed, DLO, DHI, g)
        readout = rr.choice(N, READOUT, replace=False)       # random cells (all active)
    else:  # feedforward_k2 — NO inhibition, no cycles
        P.F_INH = 0.0
        builder = lambda *a: P.build_chain(a[0], 2, a[1], a[2], a[3], a[4])
        g = P.calibrate(builder, N, seed, DLO, DHI)
        net, pool = P.build_chain(N, 2, seed, DLO, DHI, g)
        L = N // 3                                            # k=2 -> 3 layers; lag 13 reads layer 2
        readout = rr.choice(np.arange(2 * L, N), READOUT, replace=False)  # the ACTIVE layer (fair)
    accs = []
    for K in K_LIST:
        accs.append(P.acc_at(net, N, pool, CUE, readout, K, LAG, WINDOW, nt, seed + 10))
    return accs


def main():
    out = {}
    for arch in ["recurrent", "feedforward_k2"]:
        km = []
        curves = []
        for seed in SEEDS:
            accs = run_arch(arch, seed)
            curves.append(accs)
            k, note = kmax(K_LIST, accs)
            km.append(k)
            print(f"  {arch:<15} seed {seed}: " + "  ".join(f"K{K}:{a:.0%}" for K, a in zip(K_LIST, accs))
                  + f"  -> K_max={k if k is None else round(k)} {note}", flush=True)
        out[arch] = dict(curves=curves, kmax=[x for x in km if x is not None])
    json.dump(out, open(os.path.join("experiments", "constructive_kmax.json"), "w"), indent=1)
    print("\n  K_max (mean over seeds where defined):")
    for arch in out:
        ks = out[arch]["kmax"]
        print(f"    {arch:<15}: K_max = {np.mean(ks):.0f}" if ks else f"    {arch:<15}: K_max undefined (extend K)")
    if out["feedforward_k2"]["kmax"] and out["recurrent"]["kmax"]:
        r = np.mean(out["feedforward_k2"]["kmax"]) / np.mean(out["recurrent"]["kmax"])
        print(f"    feedforward / recurrent K_max ratio = {r:.2f}  -> "
              f"{'feedforward MATCHES/BEATS recurrent' if r > 0.9 else 'recurrent wins here'}")


if __name__ == "__main__":
    main()
