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

## POSITIVE CODA — the gain is delay MAGNITUDE (mean), not shape; a facet of the delay-span lever

Rule A took the task from 43.2% (uniform 1–8 ms delays) to 68.2% (cue-trained). Not cue memory —
a local rule found a *longer* delay configuration than the uniform default. The converged delays had
mean 6.6 ms (vs 4.5) with ~32% piled at the 8 ms bound. That LOOKED like a "long-skewed distribution"
finding. It is not — mean and shape were confounded, and two controls separate them.

**Matched-mean control — it is the MEAN, not the shape.** Draw delays uniform on [5, 8] (mean 6.5,
matching the converged mean, but no skew, no pile-up):

| condition | mean delay | accuracy (5 seeds) |
|---|---|---|
| uniform[1,8] (default) | 4.5 | 43.2% ± 9.0 |
| converged X, direct-gen | 6.6 | 64.5% |
| cue-trained rule | 6.6 | 68.2% |
| **uniform[5,8] — matched mean, no skew** | 6.5 | **80.7% ± 4.8** |

A clean uniform at the same mean **beats the rule by 12 points**. The skew is not a feature — the
rule's 32%-at-8 pile-up is a **truncation artifact that actively hurts**.

**WITHDRAWN — the MAX_DELAY=16 → 96% number was RELAY, not memory.** The lag/span RATIO decides
whether a delay measurement captures held memory or the relayed stimulus. The original protocol read
at 2.5× the span (lag 20, span 8) deliberately, so the readout sits past a single hop. Task 3 was
already at 1.5× (lag 12, span 8); MAX_DELAY=16 with lag still 12 is **0.75×** — the window sits INSIDE
one hop, and at span 16 a single synaptic hop (delay up to 16) lands directly in [12+onset, 22+onset).
Task-0-style check (fan_out=0 control + ratio sweep):

| span | lag | ratio | fan_out=0 | recurrent |
|---|---|---|---|---|
| 8 | 12 | 1.50× | 0.1% | 49.1% |
| 8 | 20 | **2.50×** | 0.2% | **8.2%** |
| 16 | 12 | 0.75× | 0.1% | 55.1% |
| 16 | 40 | **2.50×** | 0.3% | **1.2%** |

At span 16's proper 2.5× ratio (lag 40) the recurrent net scores **1.2% (floor)**. So the 96% is
withdrawn. More soberingly: **at 2.5× the network holds almost nothing at N=2000** (1–8%); the high
accuracies all live at ratio ≤1.5×, i.e. 1–2 hops — the multi-hop memory here is shallow, and much of
what reads as "memory" is few-hop relay reach.

**What survives (matched-ratio):** at the SAME span=8, lag=12 (1.5×, ≥2-hop for both), a higher mean
delay helps — uniform[5,8] (mean 6.5) = **80.7% ± 4.8** vs uniform[1,8] (mean 4.5) = **~46%**. That
comparison holds the ratio constant, so it is not relay. Modest, genuine, magnitude not shape.

> **PROVISIONAL.** This coda has been revised three times (long-skewed shape → mean not shape → relay
> not memory). The revision rate is itself a signal that the delay-span lever has not stabilized. The
> only clean statement right now: at MATCHED lag/span ratio, higher mean delay helps a little. Whether
> raising the *span* buys real (multi-hop) memory is UNSETTLED and must be re-measured holding the
> ratio constant (report lag/span with every delay number). Do not put a "delay-span" positive in the
> write-up until that lands.

(Reproduce: `build`/`set_delays`/`evaluate` over `task3_delay_plasticity`; ratio sweep above is 3
seeds, matched-mean is 5 seeds.)

## Place in the pattern

Entry six. Recurrence, STDP-as-gain, synaptic scaling, state-space order, capacity scaling, and now
delay plasticity — six self-organisation claims, six dissolutions under matched controls. See
[[controls-dissolve-self-organization]]. The surviving core remains small, designed, and temporal:
delay lines + coincidence detection, whose one measured lever is delay *span* (not plasticity, not
scale) — [[capacity-does-not-scale-with-N]].

Reproduce: `python experiments/task3_delay_plasticity.py A --seeds 0 1 2 3 4` (and `B`).
Data: `experiments/task3_results.json`, `task3_results_B.json`.
