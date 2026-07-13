"""Phoenix spiking neural cell — foundational state + temporal leak dynamics.

This module defines the ``Cell`` neuron state object and its passive
(sub-threshold) leak dynamics. Time is the central axis of the Phoenix
architecture, so the state carries temporal fields even where they are not
yet consumed by any dynamics.

No input integration, spike generation, refractory handling, or STDP is
implemented here — only the ``NeuronState`` fields and pure exponential decay.
"""

from __future__ import annotations

import math
import random

from phoenix.monitor import ActivityMonitor
from phoenix.spike import Spike


class Cell:
    """A single leaky neuron's state and passive membrane dynamics.

    The membrane potential ``Vm`` relaxes exponentially toward ``Vrest`` with
    time constant ``tau``. The remaining fields describe thresholds, reset,
    refractory bookkeeping, and timing that later stages of Phoenix will use.
    """

    # __slots__ eliminates the per-instance __dict__. Measured: Cell 3,340 -> far
    # smaller, Synapse 1,392 -> ~232 B. HONEST LIMIT: this does NOT close the gap
    # to the million-cell target on its own (a ~5.8x saving, not ~83x). It is a
    # cheap bridge; Struct-of-Arrays remains necessary at 1M cells and is deferred
    # deliberately until the multi-cell behavior is trusted.
    #
    # Note: __slots__ forbids setting ad-hoc attributes at runtime. That is
    # intentional — such a write now fails loudly instead of silently creating a
    # shadow field. Do NOT add __dict__ back to "fix" a test; fix the test.
    __slots__ = (
        "neuron_id", "Vrest", "Vm", "Vthresh", "Vreset", "tau",
        "refractory_period", "refractory_until", "tau_homeostasis",
        "target_rate_hz", "Vthresh_min", "Vthresh_max", "t", "last_spike_time",
        "stdp_history_window", "spike_history", "input_current",
        "trace_context_source", "trace_context_time", "spontaneous_noise_std",
        "spontaneous_silence_threshold_ms", "_rng_seed", "_rng",
    )

    def __init__(
        self,
        neuron_id: int,
        Vrest: float = -75.0,
        Vthresh: float = -50.0,
        Vreset: float = -75.0,
        tau: float = 20.0,
        refractory_period: float = 2.0,
        tau_homeostasis: float = 10000.0,
        target_rate_hz: float = 5.0,
        Vthresh_min: float = -55.0,
        Vthresh_max: float = -40.0,
        stdp_history_window: float = 100.0,
        spontaneous_noise_std: float = 1.0,
        spontaneous_silence_threshold_ms: float = 200.0,
        rng_seed: int | None = None,
    ) -> None:
        # Identity — every cell must be identifiable, e.g. for spike origin.
        self.neuron_id: int = neuron_id

        # Membrane potential starts at rest.
        self.Vrest: float = Vrest
        self.Vm: float = Vrest

        # Threshold / reset levels (unused until spike logic exists).
        self.Vthresh: float = Vthresh
        self.Vreset: float = Vreset

        # Leak time constant (ms).
        self.tau: float = tau

        # Refractory bookkeeping. Option A: hard rejection — while
        # self.t < self.refractory_until, spiking is fully suppressed and
        # incoming input is dropped. See check_threshold/receive_input.
        self.refractory_period: float = refractory_period
        self.refractory_until: float = 0.0

        # Homeostatic threshold adaptation. tau_homeostasis is deliberately
        # far slower than tau (membrane leak) so adaptation stays invisible
        # on the timescale of individual spikes and only acts as a slow
        # drift over hundreds of spikes. See apply_homeostasis.
        self.tau_homeostasis: float = tau_homeostasis
        self.target_rate_hz: float = target_rate_hz
        self.Vthresh_min: float = Vthresh_min
        self.Vthresh_max: float = Vthresh_max

        # Temporal state.
        self.t: float = 0.0
        self.last_spike_time: float = -math.inf

        # Bounded spike history for multi-pairing STDP (see check_threshold
        # and TwoCellNetwork.step). Kept alongside, not instead of,
        # last_spike_time — that field still drives refractory logic and
        # is untouched by this. History is pruned by TIME, not by count:
        # a count-based cap could silently drop a spike that's still
        # within a synapse's relevant STDP window during a high-rate burst,
        # while a time-based window directly mirrors the physical fact that
        # STDP contributions past a few tau_stdp are negligible regardless
        # of how many spikes occurred in between.
        self.stdp_history_window: float = stdp_history_window
        self.spike_history: list[float] = []

        # Synaptic input accumulator. Kept separate from Vm rather than
        # applied directly (no `Vm += w`) so that a future dendritic
        # compartment can sit between synaptic current and the soma without
        # requiring a refactor here.
        self.input_current: float = 0.0

        # trace_context: identity of the most recently *accepted* input's
        # source (a synapse's pre_id), not a blended/averaged signal across
        # sources. This is the minimal addition that lets a future network
        # layer ask "did the same source recur?" rather than only "did
        # input recur at a similar time?" Meaningful only when callers pass
        # source_id (e.g. Synapse-originated delivery); direct/manual
        # receive_input() calls that omit it simply leave it None.
        #
        # KNOWN LIMIT (not fixed): trace_context is SINGLE-SOURCE. When two
        # synapses deliver in the SAME tick, Vm sums both correctly (verified:
        # two 10 mV bumps -> -55.02 mV), but trace_context keeps only the LAST
        # source written — the other is silently lost. In a dense network,
        # where many inputs land per tick, this makes trace_context nearly
        # meaningless as a "who caused this" signal. It answers "did source X
        # deliver most recently", not "which sources contributed".
        self.trace_context_source: int | None = None
        self.trace_context_time: float | None = None

        # Spontaneous activity: sub-threshold intrinsic membrane noise
        # injected only during detected silence (see
        # maybe_spontaneous_activity). A dedicated random.Random instance
        # (not the global random module) keeps draws reproducible and
        # isolated per-cell, so tests using rng_seed are deterministic
        # regardless of what else in the process has drawn random numbers.
        self.spontaneous_noise_std: float = spontaneous_noise_std
        self.spontaneous_silence_threshold_ms: float = spontaneous_silence_threshold_ms
        # LAZY RNG. A per-cell random.Random costs 5,408 bytes — ~60% of the
        # cell's entire footprint — and Network.step() never calls
        # maybe_spontaneous_activity at all, so for every cell in a large network
        # that memory was pure waste. The generator is now created on FIRST USE.
        #
        # Determinism is untouched: the same rng_seed still produces the same
        # sequence for that cell. (A single generator SHARED across cells was the
        # obvious alternative, but it would interleave draws between cells and so
        # break the per-cell reproducibility that CellRunner's spontaneous-activity
        # tests depend on. Lazy creation gets the memory back with zero behavioral
        # risk.)
        self._rng_seed: int | None = rng_seed
        self._rng: random.Random | None = None

    def _apply_leak(self, dt: float) -> None:
        """Apply the exact exponential decay toward ``Vrest`` over ``dt``.

        Closed-form solution of ``dVm/dt = (Vrest - Vm) / tau``:

            ``Vm = Vrest + (Vm - Vrest) * exp(-dt / tau)``

        This exact-integration form makes the decay independent of step
        size. Does not advance the clock — callers own that.
        """
        self.Vm = self.Vrest + (self.Vm - self.Vrest) * math.exp(-dt / self.tau)

    def leak(self, dt: float) -> None:
        """Advance the clock by ``dt`` ms and apply pure exponential decay.

        No threshold check or spike logic is performed here.
        """
        self.t += dt
        self._apply_leak(dt)

    def receive_input(self, weight: float, source_id: int | None = None) -> None:
        """Accumulate incoming synaptic input without touching ``Vm``.

        Input is staged in ``input_current`` and only folded into the soma
        by :meth:`integrate`, keeping synaptic current separate from the
        membrane state (no direct ``Vm += weight``).

        Option A: hard rejection — while ``self.t < self.refractory_until``,
        the cell fully ignores incoming input (dropped, not accumulated).
        A future Option C (partial/reduced sensitivity during a relative
        refractory window) may relax this, but is out of scope here.

        KNOWN LIMIT (not fixed): Option A SILENTLY DISCARDS CHARGE. Input
        arriving during refractory is hard-rejected outright, not queued or
        attenuated — the charge simply vanishes, with no record that it
        arrived. In a dense network many deliveries will land inside some
        target's refractory window and be dropped invisibly, so the charge a
        cell actually integrates can be well below the charge the network
        thinks it sent.

        On acceptance, also records ``trace_context_source``/``_time`` —
        the identity and timing of the most recent contributing source.
        ``source_id`` defaults to None for backward compatibility with
        direct/manual calls that don't identify a source; rejected input
        (refractory) leaves trace_context untouched entirely, matching its
        zero effect on every other piece of state.
        """
        if self.t < self.refractory_until:
            return
        self.input_current += weight
        self.trace_context_source = source_id
        self.trace_context_time = self.t

    def integrate(self, dt: float) -> Spike | None:
        """Advance one timestep: leak, apply input, then check threshold.

        Composes the exact exponential leak with the staged synaptic input:
        the membrane first decays toward ``Vrest`` over ``dt``, then the
        accumulated ``input_current`` is added to the soma and cleared, then
        the clock advances. Finally checks whether ``Vm`` crossed threshold
        and returns the resulting :class:`~phoenix.spike.Spike`, if any.

        The membrane keeps leaking during refractory — only spike generation
        is gated (see :meth:`check_threshold`); the clock always advances by
        ``dt`` regardless of refractory state.
        """
        self._apply_leak(dt)
        self.Vm += self.input_current
        self.input_current = 0.0
        self.t += dt
        return self.check_threshold()

    def check_threshold(self) -> Spike | None:
        """Fire a spike if ``Vm`` has reached or crossed ``Vthresh``.

        On firing, records ``last_spike_time``, sets ``refractory_until``,
        resets ``Vm`` to ``Vreset``, and returns a new
        :class:`~phoenix.spike.Spike` carrying this cell's identity and the
        firing timestamp. Otherwise returns ``None``.

        Option A: hard rejection — while ``self.t < self.refractory_until``,
        spiking is unconditionally suppressed, even if ``Vm`` has somehow
        reached/exceeded ``Vthresh`` (e.g. residual elevated state right
        after a spike). This guard does not rely on ``receive_input``
        blocking alone. A future Option C (partial sensitivity during a
        relative refractory window) is planned but out of scope here.
        """
        if self.t < self.refractory_until:
            return None
        if self.Vm >= self.Vthresh:
            self.last_spike_time = self.t
            self.refractory_until = self.t + self.refractory_period
            self.Vm = self.Vreset

            self.spike_history.append(self.t)
            prune_cutoff = self.t - self.stdp_history_window
            self.spike_history = [
                ts for ts in self.spike_history if ts >= prune_cutoff
            ]

            return Spike(neuron_id=self.neuron_id, timestamp=self.t, amplitude=40.0)
        return None

    def apply_homeostasis(self, monitor: ActivityMonitor, dt: float) -> None:
        """Slowly nudge ``Vthresh`` toward a target firing rate.

        Reads the cell's recent firing rate from an external
        :class:`~phoenix.monitor.ActivityMonitor` and adjusts ``Vthresh`` by
        a step scaled by ``dt / tau_homeostasis``: firing above
        ``target_rate_hz`` raises the threshold (harder to fire), firing
        below it lowers the threshold (easier to fire). With
        ``tau_homeostasis`` far larger than ``tau``, each step is tiny
        relative to membrane dynamics — this is a slow drift, not a
        per-spike effect.

        Not called from :meth:`integrate` — the caller's simulation loop
        must invoke this explicitly alongside ``integrate``, since it
        depends on an external monitor.
        """
        current_rate = monitor.firing_rate(self.t)
        error = current_rate - self.target_rate_hz
        self.Vthresh += error * dt / self.tau_homeostasis
        self.Vthresh = max(self.Vthresh_min, min(self.Vthresh_max, self.Vthresh))

    @property
    def trace_context(self) -> tuple[int | None, float] | None:
        """(source_id, time) of the most recently accepted input, or None.

        None only until the very first input is ever accepted (checked via
        ``trace_context_time``, not ``trace_context_source`` — the source
        itself may legitimately be ``None`` for an accepted call that
        didn't identify one). This distinguishes three states: nothing has
        arrived yet (``None``), something arrived but its source is
        unknown (``(None, t)``), and a known source arrived (``(id, t)``).
        """
        if self.trace_context_time is None:
            return None
        return (self.trace_context_source, self.trace_context_time)

    def maybe_spontaneous_activity(self, monitor: ActivityMonitor) -> bool:
        """Inject sub-threshold intrinsic noise into ``Vm`` during silence.

        Only engages when ``monitor.is_silent(self.t,
        spontaneous_silence_threshold_ms)`` is True — reusing
        ActivityMonitor's existing silence detection rather than
        reinventing it. When triggered, draws a Gaussian sample and adds
        it directly to ``Vm`` (never ``input_current`` — this is intrinsic
        membrane noise, not synaptic input), then applies a defensive hard
        clamp (``Vm = min(Vm, Vthresh - 0.01)``) so a single injection can
        never itself cross threshold, regardless of ``spontaneous_noise_std``
        or how elevated ``Vm`` already was.

        Not called from :meth:`integrate` — like :meth:`apply_homeostasis`,
        this is explicit and opt-in; the caller's simulation loop must
        invoke it deliberately alongside ``integrate``.

        Scope note: this is only the primitive noise-injection mechanism.
        This primitive's higher-level purposes (replay, network-wide
        anti-freeze, path exploration enabling future planning) require a
        multi-cell network and are NOT verified by this cell-level
        implementation or its tests.
        """
        if not monitor.is_silent(self.t, self.spontaneous_silence_threshold_ms):
            return False

        if self._rng is None:
            self._rng = random.Random(self._rng_seed)
        noise = self._rng.gauss(0.0, self.spontaneous_noise_std)
        self.Vm += noise
        self.Vm = min(self.Vm, self.Vthresh - 0.01)
        return True
