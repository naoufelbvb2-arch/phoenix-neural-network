# FINDING — the delay-span lever is a feedforward delay line; recurrence contributes nothing

**Question (pre-registered, [[PREREG_delay_span_probe]]):** is the delay-span "lever" — the project's
one open positive — held multi-hop *memory*, or feedforward *relay reach*?
**Answer: relay reach.** A pure feedforward chain of the matching depth equals or beats the recurrent
net at every readout depth. The recurrence adds only interference.

## Result (N=2000, uniform[5,8] delays, rate-matched, window = 2.2×mean_delay, 3 seeds)

Depth in **mean-hops** (lag / mean_delay). Recurrent net vs the BEST acyclic feedforward chain over
k=1..4 at each depth (max over k — non-tautological):

| depth (mean-hops) | recurrent | best feedforward chain | peaking k |
|---|---|---|---|
| 1.0 | 100.0% | 100.0% | 1 |
| 1.5 | 89.7% | 100.0% | 2 |
| 2.0 | 99.9% | 100.0% | 2 |
| 2.5 | 83.2% | 100.0% | 2 |
| 3.0 | 61.9% | 99.5% | 3 |

**recurrent ≤ best-chain at every depth, all 3 seeds**, and the gap widens with depth (62% vs 99.5%
at 3 hops). The peaking chain is always the one with k ≈ depth — the signal sits at layer
k = lag/mean_delay, exactly a delay line. The recurrent net does the same thing *worse*: its loops
inject interference, not memory.

## Interpretation

- **The delay-span lever is feedforward propagation depth, not held recurrent memory.** Longer delays
  make the line longer (readable at longer lags); that is all. This is entry one ("delay lines, not
  recurrence") confirmed at the mechanism level, and it settles the last open positive.
- **Recurrence contributes nothing measurable** — the property that made this a "recurrent network"
  adds no capability a feedforward delay-line + coincidence detector lacks, and slightly hurts.
- The earlier "memory horizon scales with delay span" is consistent with this: the horizon is how far
  the feedforward wave transits before attenuating (~2–3 hops here), scaled by delay magnitude.

## What this does to the positive spine

The surviving core is now stated at its most reduced and most stable: **a designed feedforward
delay-line + coincidence detector**, with two distributional facts about how to set it up (log-normal
weights — shape; delay mean — magnitude, at matched lag/span ratio). No learning, no emergence, and
now no essential recurrence. "It's a delay line" is the answer that survives every control, so — unlike
the coda that moved three times — this one is stable.

## Scope / remaining arms (pre-registered, not yet run)

The core question is answered at N=2000. Two registered arms remain and are now secondary, since there
is no recurrent-held memory for them to scale:
- **N-scaling** (calibrated cue so 1-hop feedforward is matched across N; readout fixed) — asks whether
  feedforward propagation depth grows with N. Prediction: flattens under the calibrated control.
- **Generators at matched mean** (uniform[5,8] vs converged vs distance-derived/geometry — Task 4
  folded in) — asks whether any delay marginal beats a matched-mean uniform. Prediction: they tie;
  geometry closes as a marginal generator.

Reproduce: `python experiments/delay_span_probe.py 2000`. Data: `experiments/delay_span_core_N2000.json`.
