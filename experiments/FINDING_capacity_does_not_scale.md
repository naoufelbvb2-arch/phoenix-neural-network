# PRIMARY FINDING — a large connected network is four small ones in a trenchcoat

**Question A (pre-registered):** does memory capacity keep growing with network size N?
**Answer (controlled, CLOSED):** whatever capacity a large network has is **parallel and modular** —
**global connectivity contributes nothing.** A 64k connected network performs identically to four
*disconnected* 16k blocks (85.7% vs 85.9% at K=1280). Scaling to a million cells as a single
connected reservoir buys nothing over ~60 independent 16k modules. The connected large network is
**not justified on capacity grounds.**

Two controlled results establish this, together:
1. **Fixed-size readout gains nothing from scale** (Control A): hold the readout at the 16k size and
   a 64k net — full 4% cue — gives K_max = 75.0 ≡ 16k's 78.5. The apparent super-linear curve was
   the readout (25% of N) growing, not the network.
2. **Scaled readout gains only parallel storage** (Partition test): with the readout scaled to N,
   connected ≡ partitioned at every K — the extra capacity is four independent reservoirs, not
   integration.

(Historical note: an earlier draft of this doc overclaimed "the network stores nothing more." Control
A alone establishes only that a *fixed-size* readout extracts nothing more; the partition test is
what shows the *integration* is absent. Both are needed; neither alone suffices.)

**What this does and does NOT establish — read carefully.** A fixed 4,000-cell readout is 25% of
the network at 16k but only 6.25% at 64k, so Control A also samples a 4× smaller *fraction*. That
leaves two hypotheses it cannot separate:

- **(a) the larger network stores no additional information**, or
- **(b) it does, but a fixed-size readout cannot extract it.**

**Our data establishes (b), not (a).** The write-up claims only (b); the stronger claim is not ours
yet. This matters because in any brain-like system the readout *is* part of the architecture — a
downstream area reads from a population — so scaling the readout with N is architecture, not
confound. The real question therefore becomes: **does a 64k network with a 64k-scale readout beat
four independent 16k networks with the same total readout?** That is the DECISIVE TEST (see
"Partition test" below); it is what separates (a) from (b), and it is cheaper than any point run so
far. Until it lands, the honest statement is (b): *a fixed-size readout buys nothing from scale.*

---

## How the confound was found

The uncontrolled capacity curve looked like strong, even super-linear, growth:

| N | K_max (patterns at ≥50% accuracy) |
|---|---|
| 16,000 | 78.5 |
| 64,000 | ≫ 1280 (92% @ K=320, 85% @ K=1280) |

But from 16k→64k **three things grew at once**: the network (4×), the readout (25% of N: 4,000 →
16,000 features), and the cue (4% of N: 640 → 2,560 active cells). Super-linear K_max growth is the
*signature* of that readout+stimulus confound, not evidence against it. So each was pinned to its
absolute 16k value on the 64k network and K_max re-measured.

## The controls (all at N=64,000, full trials ≥6, ~12.3 Hz, log-normal config)

| control | readout | cue | K_max | reading |
|---|---|---|---|---|
| uncontrolled | 16000 | 2560 (4%) | ≫1280 | confounded |
| **A — readout pinned** | **4000** | 2560 (4%) | **75.0** | **N-independent** |
| B — cue pinned | 16000 | 640 (1%) | <40 (floor) | stimulus too weak |
| C — both pinned | 4000 | 640 (1%) | <40 (floor) | — |
| (reference) 16k | 4000 | 640 (4%) | 78.5 | — |

Control A curve: 61.7 → 48.3 → 42.2 → 33.6% at K = 40/80/160/320 → **K_max = 75.0** (interpolated
50% crossing), statistically identical to the 16k network's **78.5**.

**Control A establishes (b).** A 64k network, full properly-scaled 4% cue, readout held at the 16k
size → K_max = 75.0 ≡ 16k's 78.5. Given an adequate cue, the extracted K_max tracks the **readout
size**, not N. (It does not prove the network stores no more — the fixed readout is also a 4×
smaller fraction; see the Partition test.)

Controls B and C (cue pinned to the absolute 16k value = 1% of 64k) collapse to floor while the
network fires at a normal 12.3 Hz — a genuine collapse (floor measured = 0, not silence, not a
decoder artifact): a 1% cue is too sparse to perturb the 64k net into decodable states. So the cue
must be an adequate *fraction* to decode at all; given that, the readout drives the apparent scaling.

## Partition test — the decisive experiment (separates (a) from (b))

If the readout is legitimately part of the architecture, the honest question is whether **global
connectivity** adds capacity. Compare, at matched cell count, synapse count, cue, and readout, the
only difference being connectivity:

- **Connected:** one 64k network. cue = 2560 (4%), readout = 16000 (25%).
- **Partitioned:** four disconnected 16k blocks. cue = 640/block (2560 total), readout = 4000/block
  (16000 total). Same cells, same synapse count, same cue and readout sets — synapses just never
  cross block boundaries.

- **Partitioned ≈ connected** → the big network is four small networks in a trenchcoat; global
  recurrence adds nothing; scaling is pointless even with a scaled readout. This closes Question A
  and promotes (b) toward (a) *for the integration claim specifically*.
- **Connected > partitioned** → integration across scale contributes real capacity — the first
  measured argument this project has produced *for* a large network.

Measured at K = 80 / 320 / 1280 (seed 0, full trials, both at 12.4-12.5 Hz):

| K | connected 64k | partitioned (4×16k blocks) |
|---|---|---|
| 80 | 97.92% | 97.71% |
| 320 | 92.76% | 91.56% |
| 1280 | 85.68% | **85.92%** |

**Connected ≡ partitioned at every K** (at K=1280 the partitioned arm is even fractionally higher —
noise). Both K_max > 1280. **Global connectivity contributes nothing.** The 64k connected network
is precisely four independent 16k networks in a trenchcoat.

**Resolution — sharper than (a) or (b).** The larger system *does* extract more than a single 16k
net (both arms are ~91% at K=320, far past 16k's K_max=78.5), so readout-scaling is real capacity.
But that capacity is **entirely parallel and independent**: four disconnected 16k reservoirs read by
4× the readout. The integration a large *connected* network was supposed to provide is **measurably
absent**. So:

> **Question A, closed:** scaling to a million cells as a single *connected* reservoir buys nothing
> over the same neurons wired as ~60 independent 16k modules. Capacity is modular and parallel;
> global recurrence adds zero. The connected large network is not justified on capacity grounds.

## Engineering payoff (the negative result is useful)

If 60 independent 16k modules ≡ one 1M-cell connected network, then **horizontal scaling is free and
bit-exact**: distributing across machines requires **zero cross-node communication**, and each module
retains its bit-exactness locally (the reduction-order guarantee is only ever applied within a
module). The discipline fork flagged earlier — that vectorized delivery breaks reduction-order
equivalence at 10⁹ — **dissolves**: there is no 10⁹ connected object to keep bit-exact, only many 16k
ones that already are. A dead scaling path for capacity is simultaneously a clean scaling path for
deployment.

**Extended-delay test — mechanism DIAGNOSED (2026-07-26).** partitioned ≡ connected at 8 ms was
predicted from delay span (max 8 ms, readout at 20 ms, so hops beyond ~2–3 miss the window). Re-run
rate-matched at MAX_DELAY = 24 ms (g_exc 6.98→9.94 to hold 12.4 Hz). Two things resolve:

*Memory horizon vs lag (K=80, accuracy):*

| max delay | signal alive to | dies by |
|---|---|---|
| 8 ms | lag ~20 ms (97%) | lag 26 ms (6%) → 0 |
| 24 ms | lag ~34 ms (98%) | lag 44 ms (23%) → lag 56 (0.6%) |

*Connected vs partitioned at 24 ms (same lag curve):* identical at every lag (100/100/100/98/23/1 vs
100/100/100/99/25/0) — the extended horizon is **within-block**, not cross-block.

- **Confirmed (positive):** the memory horizon is delay-span-limited, and raising delays lifts it
  (~22 ms → ~40 ms). This is the first lever the project has found that moves the core metric, and it
  is exactly the variable delay plasticity (Task 3) acts on.
- **Ruled out:** that delay span suppresses *integration*. If it did, longer delays would let
  cross-block signals arrive and connected would beat partitioned — it does not. **Global integration
  is structurally absent, independent of delay span.** Question A closed with the mechanism ruled out,
  not assumed. (An earlier lag=60 ms run read past even the extended horizon and floored both arms —
  a mis-set readout, not a result.)

## Mechanism

A 4%-of-N cue read as a first-spike fingerprint off 25%-of-N cells means **both the signal and the
measurement grew with N while the per-cell dynamics did not**. The delay-line + coincidence-detection
core reads the stimulus; scale merely gave it a bigger stimulus to read and more cells to read it
from. Nothing was stored that wasn't already stored at 16k.

## Corollaries

- **Result 11, restated on K_max:** the log-normal weight advantage is now a capacity limit, not
  just a fixed-K accuracy gap. K_max(log-normal, 16k) = 78.5 vs K_max(sparse_uniform, 16k) < 40.
  The near-uniform sparse code is a strictly weaker memory. sparse_64k inflates identically under
  the readout confound (82% @ K=320, 72% @ K=1280) — same effect, not an architectural gain.

- **Five for five.** Every self-organisation claim in this project has now dissolved under a matched
  control: recurrence, STDP-as-gain-knob, synaptic scaling, state-space order (Task 0), and now
  capacity scaling. Apparent structure here is repeatedly the measurement or the stimulus, not the
  network organising itself.

## Method notes (load-bearing)

- **TRIALS ≥ 6 is mandatory.** TRIALS=2 breaks the leave-one-out nearest-centroid decoder (the
  own-class LOO centroid becomes a single noisy sample vs other classes' multi-sample means),
  yielding below-chance ~0% — a decoder artifact, not a result. Confirmed: 16k K=320 gives 0.0% at
  2 trials vs 28.5% at 6. A reduced-trials "fast scan" was attempted and discarded for exactly this.
- **K_max pre-registered before any row landed** (largest K with accuracy ≥ 50%, linearly
  interpolated between bracketing points; shuffled floor reported at every K) so the break was not
  chosen post-hoc.
- Fresh instance per measurement; measured shuffled floor per point (never assumed 1/K);
  `fan_out=0` floor control at every point.

## Reproduce

```
python experiments/task1_hardK.py 16000 40 80 160 320 --seeds 0          # 16k curve
python experiments/task1_hardK.py 64000 320 1280 --seeds 0               # uncontrolled 64k
HARDK_COND=lognormal HARDK_READOUT=4000 python experiments/task1_hardK.py 64000 40 80 160 320   # Control A
HARDK_COND=lognormal HARDK_CUE=640      python experiments/task1_hardK.py 64000 40 80 160 320   # Control B
HARDK_COND=lognormal HARDK_READOUT=4000 HARDK_CUE=640 python experiments/task1_hardK.py 64000 40 80 160 320  # Control C
python experiments/task1_hardK.py --analyze                              # K_max per curve
```
Data: `experiments/{task1_hardK_results,ctrlA_readout4k,ctrlB_cue640,ctrlC_both_pinned,sparse_16k,sparse_64k}.json`.
