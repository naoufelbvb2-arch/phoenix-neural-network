# PRE-REGISTRATION — Task 3: delay plasticity (locked before any training run)

**Question:** does *adaptive* delay structure carry **cue-specific** information a random delay
distribution does not? Per-module only — the partition result forecloses cross-module integration,
so this is a per-synapse, per-module question ([[capacity-does-not-scale-with-N]]).

## Operating point (ceiling-checked BEFORE committing seeds)

N=2000, MAX_DELAY=8 ms, frozen-random baseline sits in the 40-70% band only near the horizon edge:

| lag | K=40 | K=80 | K=160 |
|---|---|---|---|
| **12 ms** | 66.7% | 64.6% | **51.7%** |
| 16 ms | 32% | 25% | 20% |
| 20 ms | 8% | 10% | 5% |

**Task fixed at: N=2000, lag=12 ms, K=160, frozen-random = 51.7%** (dead-center; room to move both
ways; read at the horizon edge where delay sharpening could matter). First-spike fingerprint, readout
25%, LOO nearest-centroid, measured floor. 5 seeds (replication is the binding constraint; N=2000
makes it cheap).

## Two rules, orthogonal cue-specificity mechanisms — BOTH post-spike-free

Neither reads the postsynaptic SPIKE. That is the design: it structurally forecloses the runaway
(stronger → more post spikes → more potentiation) that saturated weight-STDP (92% pinned at w_max).

- **Rule A — arrival-consensus (coincidence alignment).** Per training presentation, for each post
  cell j receiving ≥2 inputs in the response window, `t*_j` = median of those **input arrival
  times** (dendritic; computed whether or not j spikes). Each contributing synapse:
  `d_s ← clip(d_s + η·(t*_j − t_arrive_s), 1, 8)`. Converges arrivals toward consensus.
- **Rule B — use-dependent speedup (myelination analogue).** A synapse that contributes to a
  coincidence (post cell had ≥2 arrivals within W of each other) shortens: `d_s ← clip(d_s − η, 1, 8)`.
  Synapses not in any coincidence relax slowly toward their initial delay. Differentiation is by
  USAGE — cue-dependent (different cues recruit different pathways), and it CREATES delay diversity
  rather than erasing it.

After training, delays FREEZE and the task is evaluated.

## Four arms (each rule)

1. **cue-trained** — plasticity ON, trained on the cues.
2. **frozen-random** — random delays, never trained (baseline, 51.7%).
3. **shuffled** — cue-trained magnitudes, delays permuted across synapses (structure destroyed,
   distribution kept).
4. **random-trained** — plasticity ON, trained on RANDOM input, evaluated on cues. *The arm that
   killed weight-STDP: it isolates cue-SPECIFICITY from a generic distribution shift.*

## Criterion (fixed up front) — THREE named outcomes, not two

Confirmed (structure learned) ONLY IF **(cue-trained > random-trained) AND (cue-trained > shuffled)**,
both across ≥5 seeds. Otherwise:
- cue ≈ random-trained → a delay-DISTRIBUTION knob, not learning → **entry six**.
- cue ≈ shuffled → distribution, not structure → **entry six**.
- **cue < frozen-random → the rule DESTROYS delay-line structure** (the one mechanism that has won
  every test needs delay DIVERSITY; a rule that equalizes latencies fights it). Informative, not null.

## REGISTERED PREDICTIONS (on record before running)

- **Rule A: cue-trained ≈ random-trained.** The consensus update is unconditionally convergent —
  every contributing synapse moves toward the median regardless of which cue drove the volley, and
  delays are shared across cues, so it converges toward the cue-AVERAGED (cue-independent) arrival
  pattern. This is STDP's symmetry problem in the delay domain (STDP raised every weight regardless
  of input; this aligns every arrival regardless of input). Predicted: entry six. Plausibly also
  **cue < frozen** (consensus erases the diversity delay-lines need).
- **Rule B:** differentiation is genuinely cue-dependent and diversity-preserving, so it is the
  candidate survivor — OR it pins all active paths at d_min (the mirror failure mode).

## Mandatory diagnostics (reported every run, before AND after training)

- **Delay distribution:** mean, sd, fraction at d_min=1 and d_max=8. (Catches Rule B's d_min pinning;
  will NOT catch Rule A — consensus pins nothing at the bounds.)
- **Per-post-cell arrival-time dispersion:** sd of input arrival times inside the coincidence window,
  averaged over post cells. This is the quantity Rule A directly acts on; its COLLAPSE is the
  predicted mechanism, and the bound-fraction diagnostic cannot see it.

## Rate-matching (methodological rule 3)

Delay plasticity changes arrival timing → coincidence → firing rate. Measure the cue-trained net's
rate AFTER training; calibrate the frozen-random control's g_exc to THAT post-training rate. Report
both pre- and post-training rates. Fresh instance per measurement.

If Rule B saturates at d_min, a per-cell conserved delay budget (speed one path, slow another) is the
natural fix — NOT added pre-emptively.
