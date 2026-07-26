# FINDING — delay plasticity carries no cue-specific structure (entry six)

**Question (pre-registered, [[PREREG_task3_delay_plasticity]]):** does *adaptive* delay structure
carry **cue-specific** information a random delay distribution does not? **Answer: no** — across two
orthogonal, post-spike-free rules, 5 seeds each. Both fail the random-input control.

## Result (N=2000, K=160, lag=12 ms, frozen-random baseline ceiling-checked at 51.7%)

| rule | cue-trained | random-trained | shuffled | frozen (rate-matched) | verdict |
|---|---|---|---|---|---|
| **A** arrival-consensus | 68.2% ± 6.5 | 67.4% | 63.8% | 44.5% | **fails** |
| **B** use-dependent speedup | 32.8% ± 6.7 | 32.4% | 40.4% | 43.4% | **fails** |

Criterion (fixed up front): confirmed only if **cue > random-trained AND cue > shuffled** across ≥5
seeds. Rule A passes the joint condition in 2/5 seeds; Rule B in 0/5.

**Rule A — cue ≈ random-trained (mean gap +0.8%, sign flips seed to seed).** The registered
prediction lands: the consensus update is unconditionally convergent — every contributing synapse
moves toward the median regardless of which cue drove the volley, and delays are shared across cues,
so it converges to the cue-AVERAGED (cue-independent) arrival pattern. This is weight-STDP's symmetry
problem in the delay domain. Note the real subtlety: cue *does* beat shuffled (+4.4 mean), so delay
*structure* helps — but that structure is **not cue-specific**, because random input builds equally
useful structure. It finds a generically-better delay configuration; it does not learn the cues.
(Diagnostics confirmed the mechanism: arrival dispersion collapsed 4.4→3.8 ms, ~32% of delays pinned
at d_max — the consensus drives delays long, which extends the horizon at lag=12; that horizon
extension, not memory, is the entire gain, and it is available from any input.)

**Rule B — degrades below frozen (third outcome, named in advance).** Shortening the delays of
coincidence-contributing synapses pulls the memory horizon *in* (lag=12 sits at the edge), so cue
scores *below* frozen-random and below shuffled. No cue-specificity either (cue ≈ random-trained).
The mechanism that has won every test in this project — designed delay lines — needs delay DIVERSITY;
a rule that collapses latencies (A, toward consensus) or shortens them (B) fights it.

## Why two rules

A single-rule negative is weak evidence about delay plasticity in general. Two rules whose
cue-specificity mechanisms are orthogonal — A differentiates by arrival timing, B by usage — both
failing the random-input control makes **"delay plasticity does not carry cue-specific structure in
this architecture"** a real claim rather than a report on one formulation.

## Method (all pre-registered, none tuned post-hoc)

- **No rule reads the postsynaptic spike** — arrivals are reconstructed from the recorded spike train
  × the synapse table. This structurally forecloses the runaway (stronger → more post spikes → more
  potentiation) that saturated weight-STDP, and is why neither rule reproduced that specific pathology.
- **Random-input arm** (train on random input, evaluate on cues) is the arm that dissolved STDP;
  it is what separates cue-specific learning from a generic distribution shift. Without it, Rule A's
  68% would have read as a positive.
- Rate-matched against the POST-training rate (delays change timing → coincidence → rate); shuffled
  and random-trained are rate-matched by construction (same/similar delay distribution).
- Ceiling-checked operating point (51.7%, not saturated); 5 seeds (replication is the binding
  constraint); diagnostics reported pre+post.

## Place in the pattern

Entry six. Recurrence, STDP-as-gain, synaptic scaling, state-space order, capacity scaling, and now
delay plasticity — six self-organisation claims, six dissolutions under matched controls. See
[[controls-dissolve-self-organization]]. The surviving core remains small, designed, and temporal:
delay lines + coincidence detection, whose one measured lever is delay *span* (not plasticity, not
scale) — [[capacity-does-not-scale-with-N]].

Reproduce: `python experiments/task3_delay_plasticity.py A --seeds 0 1 2 3 4` (and `B`).
Data: `experiments/task3_results.json`, `task3_results_B.json`.
