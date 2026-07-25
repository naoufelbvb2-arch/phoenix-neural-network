# PRIMARY FINDING — the reservoir's storage does not scale with N

**Question A (pre-registered):** does memory capacity keep growing with network size N?
**Answer (controlled):** No. Per-cell storage is N-independent. The apparent capacity growth was
the **readout** (25% of N) reading a larger stimulus off a larger cell population — not the
network storing more. **Scaling this architecture buys no additional network capacity; the
million-cell network is not justified on capacity grounds.**

A negative result, and worth more than a positive would have been: it redirects the project off a
dead scaling path, measured rather than assumed.

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

**Control A is the proof.** A 64k network, driven by a full, properly-scaled 4% cue, with the
readout held at the 16k size, stores exactly what the 16k network stored. Given an adequate cue,
K_max tracks the **readout size**, not N. The network contributes nothing per cell.

Controls B and C (cue pinned to the absolute 16k value = 1% of 64k) collapse to floor while the
network fires at a normal 12.3 Hz — a genuine collapse (floor measured = 0, not silence, not a
decoder artifact): a 1% cue is too sparse to perturb the 64k net into decodable states. So the cue
must be an adequate *fraction* to decode at all; given that, the readout drives the apparent scaling.

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
