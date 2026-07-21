"""TASK 0 — does the state space carry ORDER, or was it measuring delay lines?

PRE-REGISTERED CRITERION (fixed before running, verbatim from the brief):
    "the claim survives only if deleting all recurrent synapses measurably degrades
     separation. If separation stays at 100%, it was measuring delay lines."

DESIGN. 16 cells. Three "letter" cells are stimulated in a temporal SEQUENCE; the K=6
cues are the 6 permutations of that same three-element set. Identity is therefore held
constant across cues and ONLY THE ORDER differs — this is the "ktb vs btk" question
stated as a decoding problem. Chance is 1/6 = 16.7% nominal, but the real no-information
floor is measured with label_shuffled_floor() (it sits BELOW 1/K for this decoder).

Readout at three lags relative to cue onset:
  lag= 2 ms  — DURING the sequence (which spans onset..onset+2*hop). A fan_out=0 net
               can score here purely from the direct injection response; that is not
               memory, it is the stimulus being read back. Reported to make the
               distinction visible rather than hidden.
  lag=12 ms  — past the sequence and past the 8 ms max delay span.
  lag=20 ms  — well past both.

CONDITIONS
  recurrent  : full recurrent net (the claim)
  fan_out=0  : every synapse deleted (the mandatory control)

RATE-MATCHING CAVEAT (methodological rule 3): a fan_out=0 net has no synapses, so it
CANNOT be rate-matched to the recurrent one — it sits at the 5 Hz background-drive
baseline by construction. That absence is stated rather than papered over; it is also
exactly why fan_out=0 is a control for "is there any network contribution at all",
not a rate-matched structural control.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
from phoenix_harness import (  # noqa: E402
    MAX_DELAY, accuracy, build, label_shuffled_floor, reset,
)

N = 16
LETTERS = [0, 1, 2]
HOP = 4          # ms between successive letters -> sequence spans 8 ms
LAGS = (2, 12, 20)
WINDOW = 10
TRIALS = 8
N_TICKS = 60
JITTER = 2
CUE_W = 45.0
DRIVE_W = 30.0
P_DRIVE = 0.005
SEEDS = (0, 1, 2)


def ordered_fingerprints(net, cues, seed):
    """Present each cue as a TEMPORAL SEQUENCE and take first-spike fingerprints.

    The harness's _fingerprints injects a whole cue on one tick; order requires a
    sequential presentation, so this is the one piece Task 0 supplies itself. Everything
    else (reset, decoder, floor) is the shared harness.
    """
    rng = np.random.RandomState(seed)
    X = {L: [] for L in LAGS}
    y = []
    for ci, perm in enumerate(cues):
        for _ in range(TRIALS):
            reset(net)
            onset = 5 + rng.randint(-JITTER, JITTER + 1)
            first = {L: np.full(N, np.nan) for L in LAGS}
            for tick in range(N_TICKS):
                for i in np.flatnonzero(rng.random_sample(N) < P_DRIVE):
                    net.inject(int(i), DRIVE_W)
                for position, letter in enumerate(perm):      # ORDERED presentation
                    if tick == onset + position * HOP:
                        net.inject(int(letter), CUE_W)
                fired = net.step()
                if not fired:
                    continue
                fi = np.fromiter((net._id2idx[i] for i in fired), dtype=np.int64,
                                 count=len(fired))
                for L in LAGS:
                    lo = onset + L
                    if lo <= tick < lo + WINDOW:
                        f = first[L]
                        f[fi[np.isnan(f[fi])]] = tick - lo
            for L in LAGS:
                X[L].append(np.where(np.isnan(first[L]), float(WINDOW), first[L]))
            y.append(ci)
    return X, np.array(y)


def rate_of(net, seed):
    """Steady-state Hz over a short run (16 cells; report against the 5 Hz baseline)."""
    reset(net)
    rng = np.random.RandomState(seed)
    fired_total = 0
    ticks = 3000
    for _ in range(ticks):
        for i in np.flatnonzero(rng.random_sample(N) < P_DRIVE):
            net.inject(int(i), DRIVE_W)
        fired_total += len(net.step())
    return fired_total / (N * ticks * 1e-3)


def run() -> None:
    cues = list(itertools.permutations(LETTERS))
    print(f"TASK 0 — order discrimination on {N} cells")
    print(f"  K={len(cues)} permutations of {LETTERS} (identity fixed, ORDER varies)")
    print(f"  nominal chance {1/len(cues):.1%}; real floor measured per condition")
    print(f"  sequence spans {HOP*(len(LETTERS)-1)} ms; max delay span {MAX_DELAY:.0f} ms")
    print(f"  seeds {SEEDS}, {TRIALS} trials/cue -> {len(cues)*TRIALS} samples\n")

    conditions = {
        "recurrent (fan_out=8)": dict(fan_out=8, mode="rec"),
        "fan_out=0 (CONTROL)":   dict(fan_out=0, mode="none"),
    }

    results = {c: {L: [] for L in LAGS} for c in conditions}
    floors = {c: {L: [] for L in LAGS} for c in conditions}
    rates = {c: [] for c in conditions}

    for label, kwargs in conditions.items():
        for sd in SEEDS:
            net, _ = build(N, g_exc=6.98, log_sd=2.0, seed=sd, **kwargs)
            rates[label].append(rate_of(net, seed=500 + sd))
            X, y = ordered_fingerprints(net, cues, seed=99 + sd)
            for L in LAGS:
                M = np.stack(X[L])
                results[label][L].append(accuracy(M, y))
                floors[label][L].append(label_shuffled_floor(M, y, seed=sd)[0])

    print(f"{'condition':<24} {'rate':>8}   " +
          "  ".join(f"lag={L:<2}ms (vs floor)" for L in LAGS))
    for label in conditions:
        row = f"{label:<24} {np.mean(rates[label]):6.1f}Hz   "
        for L in LAGS:
            m, f = np.mean(results[label][L]), np.mean(floors[label][L])
            row += f"{m:6.1%} ({f:5.1%})     "
        print(row)

    print("\n  per-seed detail (accuracy):")
    for label in conditions:
        for L in LAGS:
            vals = ", ".join(f"{v:.1%}" for v in results[label][L])
            print(f"    {label:<24} lag={L:<2}: {vals}")

    print("\n=== VERDICT ===")
    for L in LAGS:
        rec = np.mean(results["recurrent (fan_out=8)"][L])
        ctl = np.mean(results["fan_out=0 (CONTROL)"][L])
        rec_f = np.mean(floors["recurrent (fan_out=8)"][L])
        ctl_f = np.mean(floors["fan_out=0 (CONTROL)"][L])
        degraded = rec - ctl
        print(f"  lag={L:>2} ms: recurrent {rec:6.1%} (floor {rec_f:5.1%})   "
              f"fan_out=0 {ctl:6.1%} (floor {ctl_f:5.1%})   "
              f"deletion changes separation by {degraded:+.1%}")


if __name__ == "__main__":
    run()
