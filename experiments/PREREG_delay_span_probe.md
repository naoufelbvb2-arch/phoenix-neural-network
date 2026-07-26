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

## Memory depth, MEASURED (graded feedforward chains — the binary→quantitative upgrade)

"recurrent > 1-hop feedforward" only shows depth > 1. Instead build explicit acyclic feedforward
chains of depth k = 1, 2, 3, 4 (disjoint layers A→B→C→…, no cycles) with the same delay generator and
rate, and measure accuracy-vs-lag for each and for the recurrent net. **Memory depth = the smallest
chain depth k whose accuracy-vs-(lag/mean_delay) curve matches the recurrent net's** (matches at the
depth where recurrent's curve cliffs). That k is the held memory depth, in hops, measured not inferred.

## N-scaling — with the capacity confound CONTROLLED

The apparent deeper horizon at 64k came from the capacity runs' readout=16000 / cue=2560 vs N=2000's
500 / 80 — a 32× difference in features and stimulus, exactly what Control A/B showed inflates results
on their own. So the decisive N arm holds **readout = 500 cells and cue = 80 cells ABSOLUTE across all
N ∈ {2000, 16000, 64000}**. Measure whether the matching chain depth grows with N.
- **Validity gate (mandatory):** at each N the absolute cue must produce decodable signal at 1 mean-hop
  (short lag). If it is at floor there (cue washed out at large N — the Control B effect), that N point
  is stimulus-limited, not memory-limited, and is reported as such, not as "shallow memory". Either way
  it answers the practical question: does a bigger net hold more from the SAME input?
- A fraction-scaled arm (readout=N/4, cue=N/25) is run alongside for contrast ONLY — it is the
  confounded version and is labeled as such.

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
2. **The absolute-fixed N arm FLATTENS the scaling** — matching chain depth does not grow with N once
   readout and cue are held absolute (and the large-N points likely hit the validity gate: cue=80
   washed out). "Memory depth scales with N" is entry seven waiting to happen; the confounded
   fraction-scaled arm will *look* like it scales, and that gap is the finding.
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
