# PhoenixNeuron v1.0 — Cell Spec

Canonical reference for the frozen single/two-cell schema, covering
`phoenix/cell.py` (`Cell`), `phoenix/synapse.py` (`Synapse`),
`phoenix/monitor.py` (`ActivityMonitor`), and `phoenix/spike.py` (`Spike`).
Accurate to the code as of v1.0 (136/136 tests passing) — not aspirational.

## Physics

| Name | Type | Purpose |
|---|---|---|
| `Cell.Vm` | `float` | Membrane potential (mV), starts at `Vrest` |
| `Cell.Vrest` | `float` | Resting potential (mV), default `-75.0` |
| `Cell.Vreset` | `float` | Potential `Vm` is reset to after a spike, default `-75.0` |
| `Cell.Vthresh` | `float` | Spike threshold (mV), default `-50.0`; adapted by homeostasis |
| `Cell.tau` | `float` | Membrane leak time constant (ms), default `20.0` |
| `Cell.refractory_period` | `float` | Minimum time (ms) after a spike before another can fire |
| `Cell.refractory_until` | `float` | Absolute time before which spiking (and input acceptance) is suppressed |
| `Cell._apply_leak(dt)` | method | Exact-exponential decay of `Vm` toward `Vrest`; does not advance the clock |
| `Cell.leak(dt)` | method | Advances the clock and applies decay only — no input/spike logic |
| `Cell.integrate(dt)` | method → `Spike \| None` | Leak, fold in `input_current`, advance clock, check threshold |
| `Cell.check_threshold()` | method → `Spike \| None` | Fires a `Spike` if `Vm >= Vthresh` and not refractory; resets `Vm`, sets `refractory_until` |
| `Synapse.delay` | `float` (property) | `distance / propagation_speed` — physical propagation time (ms) |
| `Synapse.effective_weight()` | method → `float` | `weight * exp(-distance / decay_constant)` — cable-attenuated signal |
| `Synapse.propagate(spike)` | method → `(float, float)` | Computes `(arrival_time, effective_weight)`; does not deliver input itself |

## Time

| Name | Type | Purpose |
|---|---|---|
| `Cell.t` | `float` | Internal clock (ms), advanced by `leak()`/`integrate()` |
| `Cell.last_spike_time` | `float` | Timestamp of the most recent spike (`-inf` if none); drives refractory logic |
| `Cell.spike_history` | `list[float]` | Recent spike timestamps, pruned by **time** (`stdp_history_window`), not count |
| `Cell.stdp_history_window` | `float` | Retention window (ms) for `spike_history`; diagnostic only as of v1.0 (no longer read by `TwoCellNetwork`'s STDP path) |
| `Synapse.tau_stdp` | `float` | Decay time constant (ms) shared by STDP traces and the modulation error normalization |
| `ActivityMonitor.window_size` | `float` | Sliding time window (ms) over which firing rate is computed |
| `ActivityMonitor.firing_rate(t)` | method → `float` | Spikes/second within `[t - window_size, t]` |
| `ActivityMonitor.is_silent(t, threshold_ms)` | method → `bool` | True if no spike within the last `threshold_ms`, or no history at all |
| `ActivityMonitor.is_saturated(t, rate_hz)` | method → `bool` | True if `firing_rate(t) > rate_hz` |

## Homeostasis

| Name | Type | Purpose |
|---|---|---|
| `Cell.tau_homeostasis` | `float` | Adaptation time constant (ms), default `10000.0` — ~500x `tau` so adaptation stays invisible at spike timescale |
| `Cell.target_rate_hz` | `float` | Target firing rate (Hz) homeostasis nudges `Vthresh` toward |
| `Cell.Vthresh_min` / `Vthresh_max` | `float` | Hard clamp bounds on the adaptive threshold |
| `Cell.apply_homeostasis(monitor, dt)` | method | Nudges `Vthresh` from `ActivityMonitor.firing_rate()`; **opt-in**, not called from `integrate()` |

## Short-term Memory

| Name | Type | Purpose |
|---|---|---|
| `Cell.input_current` | `float` | Synaptic input accumulator, folded into `Vm` and cleared each `integrate()` call — kept separate from `Vm` for a future dendritic compartment |
| `Cell.trace_context_source` | `int \| None` | `pre_id` of the most recently *accepted* input's source |
| `Cell.trace_context_time` | `float \| None` | Time that source's input was accepted |
| `Cell.trace_context` | `tuple[int \| None, float] \| None` (property) | `(source, time)`, or `None` until any input has ever been accepted |
| `Synapse.pre_trace` / `post_trace` | `float` | O(1) exponential STDP eligibility traces (Song, Miller & Abbott 2000); accumulate `+1.0` per own-type spike, otherwise only decay |
| `Synapse.last_trace_update` | `float` | Internal clock for lazy trace decay |
| `Synapse._decay_traces(current_time)` | method | Decays both traces continuously, same exact-exponential form as `Cell._apply_leak` |

## Learning

| Name | Type | Purpose |
|---|---|---|
| `Synapse.weight` | `float` | Causal connection strength, clamped to `[w_min, w_max]` |
| `Synapse.w_min` / `w_max` | `float` | Hard weight bounds |
| `Synapse.learning_rate` | `float` | Mutable STDP rate, read fresh each update — not baked in, so a modulation layer can rescale it per-step |
| `Synapse.A_plus` / `A_minus` | `float` | Potentiation / depression amplitude scales (independent, since biological STDP curves are often asymmetric) |
| `Synapse.update_weight(t_pre, t_post)` | method → `float` | Direct/manual pairwise modulated STDP update; still fully supported alongside the trace path |
| `Synapse.on_pre_spike(t_pre, t_post_partner)` | method → `float` | O(1) depression update (uses `post_trace`) + increments `pre_trace` |
| `Synapse.on_post_spike(t_post, t_pre_partner)` | method → `float` | O(1) potentiation update (uses `pre_trace`) + increments `post_trace`; also feeds `record_observation` |

## Prediction

| Name | Type | Purpose |
|---|---|---|
| `Synapse.observed_delays` | `list[float]` | Recorded causal pre→post delays, FIFO-capped at `max_history` |
| `Synapse.max_history` | `int` | Cap on `observed_delays` |
| `Synapse.record_observation(t_pre, t_post)` | method | Records `t_post - t_pre` only if genuinely causal (`> 0`) |
| `Synapse.mean_observed_delay` / `future_expectation` | `float \| None` (properties) | Best current guess of the pre→post delay; `None` with zero observations |
| `Synapse.delay_variance` | `float \| None` (property) | Population variance of `observed_delays`; `None` below 2 samples |
| `Synapse.confidence` | `float \| None` (property) | `regularity * sample_factor` — temporal-regularity score, `None` below 2 samples |
| `Synapse.n0` | `float` | Sample-size discount constant for `confidence`'s Bayesian-shrinkage term |
| `Synapse.compute_prediction_error(t_pre, t_post)` | method → `float \| None` | Raw `\|actual_delay - future_expectation\|`; pure, no side effects |
| `Synapse.compute_weighted_prediction_error(t_pre, t_post)` | method → `float \| None` | `compute_prediction_error` scaled by `confidence` (falls back to raw if confidence is `None`) |
| `Synapse.compute_modulation(t_pre, t_post)` | method → `float` | Learning-rate multiplier from weighted surprise; `1.0` (neutral) with no prior expectation |
| `Synapse.m_min` / `m_max` | `float` | Modulation bounds (slowest/fastest learning) |
| `Synapse.tau_error` | `float` | Normalization constant for `weighted_error` inside `compute_modulation` |

## Exploration

| Name | Type | Purpose |
|---|---|---|
| `Cell.spontaneous_noise_std` | `float` | Std. dev. (mV) of sub-threshold noise injected per triggering call |
| `Cell.spontaneous_silence_threshold_ms` | `float` | Silence duration (via `ActivityMonitor.is_silent`) required to trigger |
| `Cell._rng` | `random.Random` | Dedicated per-cell RNG (seedable via `rng_seed`), never the global `random` module |
| `Cell.maybe_spontaneous_activity(monitor)` | method → `bool` | Opt-in Gaussian nudge to `Vm` (never `input_current`) during detected silence; hard-clamped below `Vthresh` |

## Event object

| Name | Type | Purpose |
|---|---|---|
| `Spike.neuron_id` | `int` | Identifies which cell fired |
| `Spike.timestamp` | `float` | Firing cell's `t` at emission |
| `Spike.amplitude` | `float` | Fixed at `40.0` (mV); present as a field for future use |

## Two-cell integration harness

`phoenix/network.py`'s `TwoCellNetwork` (`cell_a`, `cell_b`, `synapse`, `dt`,
`current_time`, `step()`, `run()`, `current_weight`) is a deliberately minimal,
one-directional (a→b) simulation loop used to validate the above in an
integrated setting. It is **not** part of the frozen single-cell schema
itself — it will be superseded by N-cell graph architecture in the next
phase, not extended in place.

## Known simplifications

- **Refractory: Option A only.** Hard rejection — input and spiking are
  fully suppressed while `t < refractory_until`. Option C (graded/partial
  sensitivity during a relative refractory window) was discussed and
  deliberately deferred, not implemented.
- **STDP trace decay has no hard cutoff.** Unlike the earlier O(k)
  history-scan version (which excluded anything beyond `3 * tau_stdp`),
  `pre_trace`/`post_trace` decay continuously forever. An old spike
  contributes a technically-nonzero but vanishingly small term — verified
  negligible at ~4.5e-5 relative magnitude for a spike ~10×`tau_stdp` in the
  past (`exp(-10) ≈ 4.54e-5`).
- **`trace_context` tracks only the single most recent input source**, not a
  blended history of multiple recent sources — sufficient to ask "did the
  same source recur?" but not "which sources contributed recently?"
- **`eligibility_traces` (full three-factor learning with delayed global
  reward signals) was explicitly NOT implemented.** Deferred until a genuine
  delayed global signal exists to justify it (e.g. a future Guardian-style
  external signal) — `Synapse.pre_trace`/`post_trace` are STDP eligibility
  traces for the two-factor (pre/post) rule only, not three-factor.
- **`spontaneous_activity`'s higher-level purposes are unverified at this
  scale.** Replay, network-wide anti-freeze, and path exploration enabling
  future planning all require a multi-cell network to even be meaningful to
  test; only the sub-threshold noise-injection mechanism itself is verified
  here.
