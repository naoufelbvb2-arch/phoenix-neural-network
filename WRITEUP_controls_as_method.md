# In a temporal spiking network, what works is designed and distributional — not learned or emergent

**Thesis.** Across six independent "self-organisation" claims in a temporal spiking neural network
(Phoenix), every one dissolved under a matched control. Five of the six produced a *promising number
first* and survived only until the right control was added. What remains after controlling is a small,
designed, temporal core — delay lines plus coincidence detection — and two *distributional* facts
(log-normal weights, long-skewed delays) that are better generated a priori than produced by any
learning rule. The controls are the method; the surviving core is the result; and the fact that most
claims looked positive before controlling is the argument for the method.

---

## 1. The method: a matched control is not optional — it is the experiment

Each claim below began as an observed effect: a number well above chance, or a structure that appeared
to emerge from the network's own dynamics. In every case the effect was real *as a number* and false
*as an explanation*. The matched control — a condition identical in every respect except the one the
claim attributes causation to — is what separated the two. A recurring failure mode made this
essential: the natural "destroy-the-structure" control (shuffle) is too weak. It was the **stricter**
controls — rate-matching, a random-input training arm, network partitioning, direct a-priori
generation — that did the work. Where a shuffle control would have certified a positive, the stricter
control dissolved it.

## 2. Six claims, six dissolutions

| # | Claim (emergence / learning) | Promising number | Matched control that dissolved it |
|---|---|---|---|
| 1 | Recurrent computation | state-space separation | rate/structure-matched feed-forward = delay lines, not recurrence |
| 2 | STDP learns a useful gain | 92% "pinned at w_max" | random-input arm: training on noise scored identically; the 92% was the w_max clamp |
| 3 | Synaptic scaling shapes weights | a useful weight spread | the spread is better DRAWN a priori (log-normal) than produced by the controller |
| 4 | State space encodes order | 100% order separation | delete all recurrent synapses → separation unchanged; it was reading delay lines |
| 5 | Capacity scales with network size | super-linear K_max growth | pin the readout / partition into blocks → a connected 64k net ≡ four disconnected 16k blocks |
| 6 | Delay plasticity learns cue structure | 43%→68% from a local rule | random-input arm: cue-trained ≈ random-trained; the gain is a delay DISTRIBUTION, not learning |

Each is documented with data and a pre-registration where applicable (`experiments/FINDING_*.md`,
`experiments/PREREG_*.md`). Two are worth stating in full because the stricter control reversed a
result the shuffle control would have passed:

**Capacity (5).** 16k→64k, memory capacity (K_max, the number of patterns held at ≥50% accuracy)
grew super-linearly — apparently a strong argument for scale. But 16k→64k grew three things at once:
the network, the readout (25% of N), and the cue (4% of N). Holding the readout at the 16k size
returned K_max to the 16k value; and a connected 64k network scored **identically** to four
*disconnected* 16k blocks (85.7% vs 85.9% at K=1280) with matched cells, synapses, cue and readout.
Global connectivity contributes nothing: the large network is four small ones in a trenchcoat.
Capacity that exists is parallel and modular, not integrative.

**Delay plasticity (6).** A local, post-spike-free delay rule took a memory task from 43% (uniform
delays) to 68%. It beats the shuffle control (delay structure helps), so a shuffle-only design would
have called it the project's first survivor. The **random-input arm** — train the same rule on random
input, evaluate on the cues — scored the same (67%). The structure is real but not *cue-specific*; the
rule finds a generically better delay configuration. Confirmed across two orthogonal rules (arrival-
consensus and use-dependent speedup) and 5 seeds each.

## 3. What survives: a designed temporal core + two distributional facts

Controlling away the emergence claims leaves a coherent, positive picture.

- **The computational core is a FEEDFORWARD delay line + coincidence detection** — designed, not
  emergent. The delay-span lever is **propagation depth, not held recurrent memory.** Two measurements
  fix its precise status, and they must be read together:
  - vs a **1-hop** delay line, recurrence wins: past one hop the 1-hop line is at chance (20.0% ±0.0)
    while the recurrent net holds 40.4% ±8.8. So **recurrence does buy reach beyond one hop.**
  - vs **multi-hop feedforward chains** (k=1..4, matched depth), recurrence loses: a designed chain of
    the matching depth equals or beats the recurrent net at every readout depth (100% vs 62% at 3
    hops, rate-matched, depth in mean-hops, 3/3 seeds).
  - Reconciled: **recurrence is an inefficient GENERATOR of multi-hop depth.** A random recurrent
    topology produces multi-hop paths, but noisily and at a stability cost; an explicit chain delivers
    the same depth deterministically, with none of it. Recurrence is not doing something feedforward
    can't — it is doing the *same* thing, worse. The lever is delay-line depth, best obtained by design.

- **One SHAPE fact and one MAGNITUDE fact, both better a priori than learned:**
  - **Log-normal weights (shape).** Matched on *mean* weight throughout, so it is genuinely the
    spread that carries it: the weight distribution the synaptic-scaling controller was credited with
    is reproduced — better — by drawing weights log-normal at build time.
  - **Delay mean (magnitude) — provisional.** At a *matched* lag/span ratio (both span 8, lag 12), a
    uniform range at a higher mean (uniform[5,8], mean 6.5) scores 80.7% vs the default uniform[1,8]
    (mean 4.5) at ~46% — genuine, and magnitude not shape (the trained rule's apparent "long-skewed
    distribution" was a truncation pile-up at the bound). But a larger claim — that lifting the span
    keeps raising accuracy — was WITHDRAWN as a relay artifact: at the proper 2.5× ratio it is floor.
    The settled part is small (higher mean helps a little at matched ratio); the span lever is unsettled.

The pattern: what works is **designed and distributional** — a shape result (log-normal weights) and a
magnitude result (delay mean at matched ratio), neither learned — read out by a **feedforward delay
line** that needs no recurrence. Learning, emergence, and even recurrence, wherever claimed, were the
measurement or the stimulus in disguise.

## 4. The controls, as a reusable toolkit

- **Rate-matching** — match firing rate between conditions (recompute gain if a manipulation changes
  it); rate differences masquerade as structure. Match to the *post-manipulation* rate.
- **Random-input arm** — train the learning rule on random input, evaluate on the task. Separates
  task-specific learning from a generic parameter drift. Dissolved claims 2 and 6.
- **Partition / disconnection** — to test whether global structure matters, compare against the same
  units with the structure removed (disconnected blocks). Dissolved claim 5's integration reading.
- **Direct a-priori generation** — if a rule "learns" a configuration, draw that configuration
  directly and see if the gain survives without the rule. Turned claims 3 and 6 into distributional
  positives.
- **Pre-registration** — fix the operating point, criterion, and predictions before running; the
  break/threshold must not be chosen after seeing the curve.
- **Replication (≥5 seeds)** — the binding constraint; several point estimates reversed across seeds.
- **Fresh instance per measurement; measured floor (never assumed 1/K); watch readout dimensionality
  AND stimulus size** — both scale with N and both inflate results (the capacity confound).

## 5. Engineering corollary (the negative is useful)

If a connected 1M-cell network ≡ ~60 independent 16k modules, then horizontal scaling is **free and
bit-exact**: distribution across machines needs zero cross-node communication, and each module keeps
its numerical (reduction-order) guarantees locally. A dead scaling path for capacity is a clean
scaling path for deployment.

## 6. The architectural consequence — the stability apparatus is unnecessary

If recurrence is an inefficient way to get depth that design supplies directly, then everything the
recurrence *cost* — measured this session — becomes optional. Recurrence produced: bistability with a
zero-width operating band; metastable ignition after ~3000 ticks; homeostasis failing three ways;
synaptic scaling that stabilised but damaged function; inhibition required only to stop runaway;
per-spike delivery-order complexity in the event heap.

A feedforward delay line has **none** of these. No cycles ⇒ no runaway ⇒ no ignition ⇒ **no
homeostatic controller, no inhibition-for-stability, provable termination, trivial parallelism, and a
firing rate that is a direct function of the input rather than an unstable equilibrium to be
regulated.** The negative result on recurrence is therefore a large *positive* on architecture: the
same computation, obtained by design, discards the entire stability apparatus the recurrent version
needed. This is stated as implied by the measurements; the **constructive test** (§7) is what turns it
from implication into demonstration.

## 7. Constructive test — build the implied architecture, run every benchmark

Everything above is negative (X is not doing the work). The constructive form: build the architecture
the analysis implies — an explicit feedforward delay-line + coincidence-detector network, **no cycles,
no inhibition, no homeostasis, no plasticity**, log-normal weights, matched mean delay — and run it
against the recurrent architecture on the repo's benchmarks.

**Result: it matches.** On capacity (K_max), read fairly — each architecture from its own most-active
cells — the two are equal (~98–100% at K=1280, floor 0). On memory depth it equals or beats the
recurrent net (99.5% vs 62% at 3 mean-hops, where the recurrent signal decays into interference). And
it is far more robust to readout: with a *random* readout the feedforward net still holds 97% while the
recurrent net drops to 35%, because feedforward concentrates the response in a layer while recurrence
disperses and mixes it. So the recurrence is not just inessential — where it differs it *hurts*, and
its capacity is no higher.

**An honest correction, kept in the record because it is the point:** a first pass showed feedforward
"beating" recurrent 6× on capacity. That was a readout-selection artifact — feedforward read from its
concentrated active layer, recurrent from random cells (Control A, again). Giving the recurrent net its
own best cells closed the gap. The claim is **match**, not beat; the inflated number was controlled
away — the method applied to the paper's own positive result.

The constructive positive therefore stands in its defensible form: the same computation, obtained by
design, reproduces the recurrent architecture while discarding the entire stability apparatus (§6).

## 8. Scope and honesty

These are results about *this* architecture (rate-coded readout, first-spike fingerprints, this delay/
weight regime), not about spiking networks in general. The claim is not "learning cannot work in
SNNs"; it is that in this system, six specific emergence claims did not survive matched controls, and
the surviving mechanisms are distributional and designed. The method — and the discipline of treating
a promising number as a hypothesis about a confound until a matched control rules it out — is the
transferable part.

---

*Source: this repository. Findings and pre-registrations in `experiments/FINDING_*.md` and
`experiments/PREREG_*.md`; data in the committed `*_results.json`; each result reproducible via the
commands in its finding doc.*
