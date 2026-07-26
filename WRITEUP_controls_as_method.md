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

- **The computational core is delay lines + coincidence detection** — designed, not emergent. It is
  what every dissolved claim turned out to be "really" doing. Its candidate lever is the **delay
  span** (raising max delay appears to extend the memory horizon), but this is **UNSETTLED**: the
  lag/span RATIO decides whether a readout captures held memory or the relayed stimulus, and the
  horizon numbers were read at *different* ratios. Whether span buys real multi-hop memory or just
  extends few-hop relay reach must be re-measured holding the ratio constant. At a strict 2.5× ratio
  the network holds little (1–8% at N=2000) — the multi-hop memory is shallow.

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

The pattern: what works is **designed and distributional** — a shape result (log-normal weights) and,
provisionally, a magnitude result (delay mean at matched ratio), neither learned. The delay-span lever
is the project's one open positive and must be re-measured at fixed ratio before it is claimed.
Learning and emergence, wherever claimed, were the measurement or the stimulus in disguise.

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

## 6. Scope and honesty

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
