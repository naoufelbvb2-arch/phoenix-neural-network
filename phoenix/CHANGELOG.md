# Phoenix — Changelog

Build history of `PhoenixNeuron` v1.0, in the order it was actually implemented.
Test counts are the real, verified totals from each step's full test-suite run
(`pytest tests/`), not rounded estimates.

## 1. NeuronState + exact-exponential leak — 7 tests

`phoenix/cell.py`: `Cell` constructor (`Vm`, `Vrest`, `Vthresh`, `Vreset`, `tau`,
`refractory_period`, `refractory_until`, `t`, `last_spike_time`) and `leak(dt)`,
implementing the closed-form solution `Vm = Vrest + (Vm - Vrest) * exp(-dt/tau)`
rather than a discrete Euler step — decay is exact and independent of step size
(verified by a dt-invariance test: one `leak(dt=10)` call equals ten
`leak(dt=1)` calls).

## 2. Input integration — 6 new tests (13 total)

Added `input_current` (accumulator, not applied directly to `Vm`) and
`receive_input(weight)` / `integrate(dt)`. Synaptic input is staged separately
from the membrane state so a future dendritic compartment can sit between
synaptic current and the soma without a refactor.

## 3. Threshold + Spike object — 10 new tests (23 total)

`phoenix/spike.py`: `Spike` dataclass (`neuron_id`, `timestamp`,
`amplitude=40.0`) — a first-class event object, not a boolean or float, since
spike timing/identity matters for STDP and synchrony later. `Cell.check_threshold()`
fires a `Spike`, resets `Vm` to `Vreset`, and `integrate()` now returns
`Spike | None`.

## 4. Refractory dynamics (Option A: hard rejection) — 7 new tests (30 total)

After a spike, `refractory_until = t + refractory_period`. While
`t < refractory_until`, `receive_input()` drops incoming input entirely and
`check_threshold()` unconditionally suppresses spiking, even if `Vm` is
somehow still elevated. The membrane keeps leaking during refractory — only
spike generation is gated. Option C (graded/partial sensitivity) was
discussed and deliberately deferred (see `CELL_SPEC.md`).

## 5. Activity monitoring — 8 new tests (38 total)

`phoenix/monitor.py`: `ActivityMonitor`, a passive diagnostic observer (no
side effects on any `Cell`) tracking spike timestamps over a sliding window —
`firing_rate()`, `is_silent()`, `is_saturated()`. Built before homeostasis so
a cell's stability (neither dead nor exploding) could be verified before
trusting any learning signal built on top of it.

## 6. Homeostasis — 7 new tests (45 total)

`Cell.apply_homeostasis(monitor, dt)` nudges `Vthresh` toward `target_rate_hz`
based on `ActivityMonitor.firing_rate()`, clamped to `[Vthresh_min, Vthresh_max]`.
`tau_homeostasis` defaults to 500x `tau`, keeping adaptation invisible on the
timescale of individual spikes. Not called from `integrate()` — explicit,
opt-in, same pattern later reused by `spontaneous_activity`.

## 7. Synapse: geometric delay + attenuation — 8 new tests (53 total)

`phoenix/synapse.py`: `Synapse` as a geometric-graph edge (`pre_id`, `post_id`,
`weight`, `distance`), not a cell of a dense/sparse weight matrix. `delay`
(`distance / propagation_speed`) and `effective_weight()`
(`weight * exp(-distance / decay_constant)`) are both derived from distance.
`propagate(spike)` only *computes* arrival time and attenuated weight —
delivery is a separate concern.

## 8. TwoCellNetwork: two-cell simulation loop — 7 new tests (60 total)

`phoenix/network.py`: a deliberately minimal one-directional (a→b) network.
Hybrid design — a fixed timestep drives `integrate(dt)` on both cells every
tick, while spike delivery across the synapse's delay uses a small pending-
delivery queue. Documented simplification: delivery is coarsened to the tick
whose `[t, t+dt)` interval contains the arrival time, not the exact sub-step
moment.

## 9. STDP (`update_weight`) — 9 new tests (69 total)

Classic pairwise STDP added to `Synapse`: `update_weight(t_pre, t_post)`,
potentiating for causal timing and depressing for anti-causal, with
`learning_rate` exposed as a plain mutable attribute (not baked in) so a
future reward-modulation layer could rescale it without a refactor.

## 10. Live STDP wiring into TwoCellNetwork — 5 new tests (74 total)

`TwoCellNetwork.step()` began calling `synapse.update_weight()` on qualifying
spike pairs during live simulation, not just in isolated unit tests.

## 11. Multi-pairing fix + `spike_history` — 4 new tests (78 total)

Bug fix: the original live wiring paired each new spike only against the
*other* cell's single `last_spike_time`, producing spurious updates from
stale pairings. Fixed by adding `Cell.spike_history` (time-windowed, not
count-windowed) and pairing each new spike against every entry in the other
cell's history within `3 * tau_stdp` — replacing single-last-spike-time
matching with correct multi-pairing STDP. One earlier test
(`test_causal_ordering_in_live_network_potentiates`) was superseded by a
corrected version once this fix changed the relevant math.

## 12. Prediction pathway V1: observation recording — 9 new tests (87 total)

`Synapse.observed_delays`, `record_observation(t_pre, t_post)` (records only
genuinely causal delays, FIFO-capped at `max_history`), `mean_observed_delay`,
`delay_variance` (population variance, `None` below 2 samples). Deliberately
decoupled from `update_weight()` — weight measures causal strength; this is
groundwork for a *separate* temporal-regularity measure. Lives on the synapse,
not the cell, since "does post usually follow me?" is inherently local.

## 13. Prediction pathway V2: confidence + future_expectation — 9 new tests (96 total)

`confidence = regularity * sample_factor`, where `regularity = exp(-CV)` (CV =
coefficient of variation, not raw variance, so long-delay synapses aren't
penalized for a bigger absolute spread) and `sample_factor = n / (n + n0)`
(Bayesian-shrinkage discount against a couple of coincidental observations).
`future_expectation` delegates to `mean_observed_delay` but only needs `n>=1`
(vs. `confidence`'s `n>=2`) — a single observation is still a usable, if
low-confidence, guess.

## 14. Prediction pathway V3: prediction_error — 9 new tests (105 total)

`compute_prediction_error(t_pre, t_post)` and
`compute_weighted_prediction_error(...)` (scaled by `confidence`, falling back
to the raw error when confidence is undefined). Pure, side-effect-free
computations — must be called *before* `record_observation()` for the same
event, or the comparison baseline is contaminated by the very observation
being evaluated. Only handles "actual delay mismatch"; "predicted but never
happened" is explicitly out of scope (needs a timeout mechanism that doesn't
exist yet).

## 15. Prediction pathway V4: modulation → `update_weight` — 6 new tests (111 total)

`compute_modulation(t_pre, t_post) = m_min + (m_max - m_min) * (1 - exp(-weighted_error / tau_error))`,
now scaling `learning_rate` inside `update_weight()`. A perfectly-predicted
event learns slowly (`m_min`); a surprising one learns fast (bounded by
`m_max`). Both bounds are explicit, mutable tunables — the m_min tradeoff
(slow forgetting vs. premature "freezing") is left to the caller, not
resolved in code.

## 16. Live network wiring of `record_observation` — 4 new tests (115 total)

`TwoCellNetwork.step()` began calling `record_observation()` immediately after
each `update_weight()` call for the same pair — feeding the prediction
machinery from live spike data for the first time. Call order
(`update_weight` before `record_observation`) is load-bearing: reversing it
would make a synapse compare an event against an expectation that already
includes that very event.

## 17. O(1) trace-based STDP refactor — 9 new tests (124 total)

Replaced the O(k)-per-spike history-scan multi-pairing (step 11) with the
Song, Miller & Abbott (2000) all-to-all exponential trace scheme:
`Synapse.pre_trace` / `post_trace`, decayed lazily via `_decay_traces()`, and
`on_pre_spike()` / `on_post_spike()`. Verified mathematically equivalent to
the exact multi-pair sum (not just "a decaying number") via a closed-form
analytical test. Trade-off: the old scheme's hard `3 * tau_stdp` cutoff is
gone — traces now decay continuously with no exclusion boundary, so an old
spike contributes a technically-nonzero (but exponentially negligible) term
forever. `update_weight()` was kept fully intact for direct/manual use.

## 18. `trace_context` — 6 new tests (130 total)

`Cell.receive_input(weight, source_id=None)` now also records
`trace_context_source` / `trace_context_time` on acceptance — the identity
(not a blend) of the single most recently contributing input source. Lets a
future network layer ask "did the *same* source recur?", not just "did input
recur at a similar time?" `TwoCellNetwork` passes `synapse.pre_id` through on
delivery so live traffic populates it meaningfully.

## 19. `spontaneous_activity` — 6 new tests (136 total)

`Cell.maybe_spontaneous_activity(monitor)`: opt-in, sub-threshold Gaussian
noise injected directly into `Vm` (never `input_current` — intrinsic, not
synaptic) only when `ActivityMonitor.is_silent()` reports true, with a
defensive hard clamp (`Vm = min(Vm, Vthresh - 0.01)`) guaranteeing a single
injection can never itself cross threshold. Uses a dedicated
`random.Random(rng_seed)` instance, not the global `random` module, for
reproducible tests. This is the final piece of the v1.0 spec.

---

**Final: 136/136 tests passing.** This closes PhoenixNeuron v1.0's single-cell
specification. Any further additions belong to network-level architecture
(N-cell graphs, concept emergence, language, planning), not the cell itself,
per the project's golden rule: no component is added unless it serves a clear
computational function (memory, prediction, learning, generalization,
exploration, efficiency).

## 20. Closed-loop validation harness (CellRunner) — 9 new tests (145 total)

Added after v1.0 was frozen: `phoenix/runner.py`'s `CellRunner` drives a
single `Cell` + `ActivityMonitor` through the full feedback loop
(`receive_input` → `integrate` → record real spike → `apply_homeostasis` →
`maybe_spontaneous_activity`) over a long horizon — closing a loop the v1.0
unit tests only ever exercised half at a time (moving `cell.t` by hand and
feeding a pre-fabricated monitor). New files only (`phoenix/runner.py`,
`tests/test_cell_runner.py`); 145/145 total with zero regressions and zero
existing files touched. It confirms `apply_homeostasis` reaches a genuine
**interior** equilibrium (`tau=100ms`: `Vthresh` converges `-53.0 → ~-48.4 mV`,
tail σ ≈ `0.06 mV`, clear of both bounds) rather than pinning at a limit, and
surfaces two properties: at the default `tau=20ms` the 5 Hz setpoint sits near
rheobase (~4× noisier convergence — `tau`/`target_rate_hz` must be tuned
jointly), and the realized max firing rate under saturating input is
`1000/(refractory_period + dt)` not `1000/refractory_period` (an
inject-then-integrate ordering effect shared with `TwoCellNetwork.step()`,
architectural rather than harness-specific). See `CELL_SPEC.md`'s post-v1.0
addendum for the detailed reference. This is validation tooling layered on top
of the frozen cell — the `v1.0-single-cell` tag and everything it marks are
untouched.
