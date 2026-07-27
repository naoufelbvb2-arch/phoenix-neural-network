# FINDING — the implied feedforward architecture matches the recurrent one (and is far simpler)

**Constructive test** (the positive counterpart to the six dissolutions): build the architecture the
analysis implies — an explicit **feedforward delay-line + coincidence-detector** net, **no cycles, no
inhibition, no plasticity/homeostasis**, log-normal weights, matched mean delay — and run it against
the recurrent architecture on the repo's benchmarks. **Result: it matches on capacity, equals-or-beats
on memory depth, is more robust to readout, and discards the entire stability apparatus.** No "beats
6×" claim survives control — see the withdrawal below.

## Capacity (K_max), N=2000, K=1280, rate-matched, floor-controlled

| readout | recurrent (with inhibition) | feedforward (no inhibition, no cycles) |
|---|---|---|
| **best-300 active cells (fair, both)** | 96.9–98.6% (floor 0.0%) | 100% (floor 0.0–0.1%) |
| random-300 cells | 33.7–36.8% | 97.1–97.8% |

**At a fair readout — each architecture read from its own most-active 300 cells (selected by activity,
no label leak) — the two MATCH** (feedforward marginally higher). Neither has a higher raw capacity.

### WITHDRAWN — the "feedforward beats recurrent 6× on capacity" number was a readout artifact

A first pass read feedforward from its concentrated active layer but recurrent from *random* cells,
giving feedforward K_max > 2560 vs recurrent ~381 (>6×). That was **Control A again** — a readout
selection asymmetry, not capacity. Give the recurrent net its own best-300 cells and it reaches 98.6%
at K=1280 (from 36.8% random). The 6× is withdrawn. (Caught before it reached the write-up — the method
applied to my own result.)

## The real architectural difference — readout robustness (stated carefully)

With a *random* 300-cell readout, feedforward holds 97% but recurrent only 35%. The feedforward chain
**concentrates** the cue response in a layer, so it is readable from anywhere; recurrence **disperses
and mixes** the response with interference, so a fixed/random readout barely reads it. This is a real
property (readout robustness / clean organization), NOT a raw-capacity advantage — with the best cells
selected, they match.

## Memory depth (from the delay-span probe — full readout for both, no selection asymmetry)

Feedforward ≥ recurrent at every depth, beating it at deep lags (99.5% vs 62% at 3 mean-hops) where the
recurrent signal decays into interference while the clean chain still carries it. See
[[capacity-does-not-scale-with-N]] / FINDING_delay_span_probe.

## Verdict

The feedforward delay-line architecture **matches** the recurrent one on capacity, **equals or beats**
it on memory depth, and is **more robust to readout** — while being dramatically simpler: no cycles ⇒
no runaway ⇒ no ignition ⇒ no homeostatic controller, no inhibition-for-stability, provable
termination, trivial parallelism, rate = f(input). The recurrence is not just inessential; where it
differs it **hurts** (interference), and its capacity is no higher. This is the constructive positive:
the same computation, obtained by design, discarding the stability apparatus the recurrent version
needed. It is a *match*, honestly — the inflated "beats" was a readout confound, controlled away.

Reproduce: `python experiments/constructive_test.py` (raw), plus the best-vs-random readout control
inline (see git log). Data: `experiments/constructive_kmax.json`.
