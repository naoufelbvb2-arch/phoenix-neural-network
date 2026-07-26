# PRE-REGISTRATION — the delay-span probe (locked before running)

Settles the project's one open positive: is the delay-span "lever" **held multi-hop memory** or
**single-hop relay reach**, measured in hops rather than inferred; does memory depth scale with N
*under matched stimulus/readout*; and do delay generators differ beyond their mean (Task 4 folded in)?
Prompted by the withdrawal of the 96% number as relay ([[capacity-does-not-scale-with-N]] finding;
Task 0 was the same relay trap).

## Units and matching (the corrections that make it mean anything)

- **Depth unit = lag / mean_delay (mean-hops), NOT lag / max_span.** After k hops an arrival lands at
  ≈ k × mean_delay. uniform[1,8] (mean 4.5) at lag 12 is ~2.7 mean-hops; uniform[5,8] (mean 6.5) at the
  same lag is ~1.8. Holding lag/max_span constant does NOT hold hop-depth constant across distributions.
  All conditions are compared at **fixed lag/mean_delay**, reported on every line.
- **Readout window scales with mean_delay** (window = round(2.2 × mean_delay), reported per line): a
  fixed 10 ms slit samples a proportionally narrower slice of the arrival spread as delays grow.
- **Rate-matched everywhere:** recurrent recalibrated (g_exc) per delay generator to a common rate;
  **feedforward arms recalibrated to the recurrent arm's rate** (only source cells project, so their
  synapse count and drive differ by construction).

## Memory depth, MEASURED (graded feedforward chains — quantitative, and non-tautological)

Build explicit acyclic feedforward chains of depth k = 1, 2, 3, 4 (disjoint layers A→B→C→…, no cycles)
with the same delay generator and rate. A k-hop chain first delivers at ≈ k × mean_delay, so the chain
that "matches" a given lag is largely fixed by k ≈ lag/mean_delay — searching for the *smallest
matching* chain would just re-derive the lag, not measure memory. Non-tautological version: **at each
lag, compare the recurrent net against the BEST chain over all k (max over k=1..4), not the first that
matches.**
- recurrent ≈ best-chain → pure feedforward propagation at that horizon; no held memory beyond a chain.
- recurrent > best-chain → the recurrent net holds something **no feedforward structure of any depth**
  achieves at that lag — genuine held memory.
Also report **which k peaks** at each lag — the feedforward propagation depth, useful context for
reading the recurrent number.

## N-scaling — inject the SAME signal at every N (calibrated cue), then measure propagation

The thing that must be held constant is *where the signal starts*. Neither absolute nor fraction-fixed
cue does that: absolute starts weaker at large N (0.125% at 64k), fraction starts stronger (Control A).
Fix: **calibrate cue size per N so 1-hop feedforward accuracy is matched across all N** (target 70% ±5;
readout held at 500 absolute so observation is constant). Then depth is measured from a common
baseline, and any cross-N difference in the fall-off with lag is **propagation, not injection**.
- **Report the calibrated cue size per N as a result.** If 64k needs a much larger absolute cue just to
  reach 70% at one hop, that is itself informative (and is the Control B effect, now quantified).
- This **removes** the earlier validity-gate ambiguity: with 1-hop performance equalized, no N point is
  stimulus-limited, so a flat depth-vs-N is "**can't hold more**" and is cleanly distinct from "can't
  inject" — the distinction that has been the method all session. Do NOT merge the two.
- A **fraction-scaled arm** (readout=N/4, cue=N/25) is run **alongside and reported side by side**, even
  though it is the more flattering number — it is the confounded version, and the gap between "it looks
  like it scales" (fraction) and "it flattens" (calibrated) is the finding.

## Delay generators (Task 4 folded in) — compared at MATCHED MEAN

At a fixed N (2000) and fixed depth, compare delay generators all normalized to the **same mean
delay** (≈6.5 ms): uniform[5,8], the rule-converged distribution (mean ~6.6), and **distance-derived
delays** (geometry: positions in space, delay = distance/speed, scaled to mean 6.5). Plus uniform[1,8]
as the low-mean reference.
- distance-derived > matched-mean uniform → spatial correlation carries computation beyond the mean →
  geometry (the brief's last item) is real.
- distance-derived ≈ matched-mean uniform → geometry's value is only the delay marginal it induces,
  whose mean is the lever. The brief closes with a measurement, not an assumption.

## REGISTERED PREDICTIONS (on record before running)

1. **Memory depth is shallow and single-hop-dominated at N=2000** — recurrent matches the 1- to 2-hop
   chain, not deeper.
2. **The calibrated N arm FLATTENS the scaling** — with 1-hop performance equalized across N (cue
   calibrated, readout fixed), the recurrent-vs-best-chain gap does not grow with N: a bigger net holds
   no more hops from an equal-strength input. "Memory depth scales with N" is entry seven waiting to
   happen. The confounded fraction-scaled arm will *look* like it scales; the two are reported side by
   side and that gap is the finding. (Secondary prediction: 64k needs a much larger calibrated cue to
   reach 70% at 1 hop — the Control B washout, quantified.)
3. **Generators match at matched mean** — distance-derived ≈ uniform[5,8]; spatial correlation adds
   nothing beyond the mean. Geometry closes as a delay-marginal generator.

Given five of the last six results, outcomes 2 and 3 are the expected ones, and predicting them in
advance is worth more than explaining them after. A surviving positive (depth grows with N under the
absolute control, or geometry beats matched-mean uniform) would be the project's first, and the design
is built to make either verdict clean.

## Method

Fresh instance per measurement; measured shuffled floor per point; rate-matched per above; ≥5 seeds;
lag/mean_delay AND window reported on every line. Operating accuracies kept off ceiling and off floor
(adjust K per N so the recurrent 1-hop point is mid-band, ceiling-checked first as in Task 3).
