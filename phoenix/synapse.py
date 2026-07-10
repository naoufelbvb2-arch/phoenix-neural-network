"""Synaptic connections between Phoenix cells — a geometric graph, not a matrix.

Each connection is its own object carrying weight, geometric distance, and
the propagation delay and attenuation derived from that distance. Modeling
connectivity this way (rather than as a dense/sparse weight matrix indexed
by cell) keeps the network spatially grounded, which later cable-theory and
concept-organization work depends on.
"""

from __future__ import annotations

import math

from phoenix.spike import Spike


class Synapse:
    """A single directed connection from a presynaptic to a postsynaptic cell.

    Distance drives two derived physical quantities: propagation ``delay``
    (how long a spike takes to arrive) and attenuation of the signal's
    ``effective_weight`` (how much charge survives the cable over that
    distance).
    """

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        weight: float,
        distance: float,
        propagation_speed: float = 1.0,
        decay_constant: float = 10.0,
        learning_rate: float = 0.01,
        tau_stdp: float = 20.0,
        A_plus: float = 1.0,
        A_minus: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 20.0,
        max_history: int = 50,
        n0: float = 5.0,
        m_min: float = 0.1,
        m_max: float = 2.0,
        tau_error: float = 20.0,
    ) -> None:
        self.pre_id: int = pre_id
        self.post_id: int = post_id
        self.weight: float = weight
        self.distance: float = distance
        self.propagation_speed: float = propagation_speed
        self.decay_constant: float = decay_constant

        # STDP parameters. learning_rate is intentionally a plain mutable
        # attribute (read fresh at each update_weight() call, not baked in)
        # so a future reward-modulation layer can rescale it per-step
        # without any refactor here — see the reward-modulated STDP step.
        self.learning_rate: float = learning_rate
        self.tau_stdp: float = tau_stdp
        self.A_plus: float = A_plus
        self.A_minus: float = A_minus
        self.w_min: float = w_min
        self.w_max: float = w_max

        # Prediction pathway V1: pure recording of observed pre->post delays.
        # This lives on the synapse (not the cell) because "if I fire, does
        # post usually follow, and after how long?" is inherently a local,
        # per-connection question — consistent with Phoenix's
        # no-central-representation principle. Deliberately independent of
        # weight/STDP: weight measures causal strength, this groundwork will
        # (in V2) measure temporal regularity, which a weak-but-consistent
        # synapse can score highly on despite low weight.
        self.observed_delays: list[float] = []
        self.max_history: int = max_history

        # V2: sample-size discount constant for `confidence`, exposed as a
        # plain mutable attribute (not baked in) — consistent with how
        # learning_rate is externally adjustable.
        self.n0: float = n0

        # V4: STDP modulation bounds. m_min/m_max are explicit tunables, not
        # hardcoded — the tradeoff between them (higher m_min risks slow
        # drift/forgetting of stable patterns; lower m_min risks the synapse
        # "freezing" once it locks onto a pattern) is left to the caller,
        # not resolved here. tau_error reuses tau_stdp's default since both
        # normalize a time-scale quantity, absent a reason to differ.
        self.m_min: float = m_min
        self.m_max: float = m_max
        self.tau_error: float = tau_error

        # O(1)-per-spike exponential STDP traces (Song, Miller & Abbott
        # 2000 all-to-all scheme), replacing history-scan multi-pairing.
        # Each trace accumulates +1.0 on its own-type spike and otherwise
        # only decays continuously — never reset to a fixed value — which
        # is what makes trace(t_n) mathematically equal to the exact sum
        # sum_i exp(-(t_n - t_i)/tau_stdp) over every past same-type spike,
        # not merely "some decaying number." last_trace_update is this
        # pair of traces' internal clock, decayed lazily on each spike
        # event rather than every simulation tick (same lazy-decay pattern
        # as Cell's leak, just invoked from spike events instead of dt).
        self.pre_trace: float = 0.0
        self.post_trace: float = 0.0
        self.last_trace_update: float = 0.0

    @property
    def delay(self) -> float:
        """Time (ms) for a spike to physically travel this synapse's distance."""
        return self.distance / self.propagation_speed

    def effective_weight(self) -> float:
        """Base weight attenuated exponentially by cable distance."""
        return self.weight * math.exp(-self.distance / self.decay_constant)

    def propagate(self, spike: Spike) -> tuple[float, float]:
        """Compute when and how strongly a spike arrives at the postsynaptic cell.

        Does not deliver input to any cell — delivery/scheduling belongs to
        a future event-queue/simulation-loop step.
        """
        arrival_time = spike.timestamp + self.delay
        return arrival_time, self.effective_weight()

    def update_weight(self, t_pre: float, t_post: float) -> float:
        """Apply one prediction-error-modulated STDP update.

        ``delta_t = t_post - t_pre``: positive means pre fired before post
        (causal — potentiate), negative means pre fired after post
        (anti-causal — depress). Exactly zero is treated as neither
        causal nor anti-causal and produces no change, since there is no
        well-defined ordering to reinforce or punish.

        The raw ``learning_rate`` is scaled by :meth:`compute_modulation`
        before computing ``dw`` — a well-predicted event learns slowly
        (modulation near ``m_min``), a surprising one learns fast
        (modulation near ``m_max``). For a fresh synapse with no prior
        expectation, modulation is neutral (1.0), so this degrades to
        plain unmodulated STDP.

        Call-order dependency: like :meth:`compute_prediction_error`, the
        modulation computed here reflects ``future_expectation`` as it
        stood *before* this event. This method never calls
        :meth:`record_observation` itself, so that's naturally satisfied
        as long as callers don't record the observation before calling
        this for the same event.
        """
        delta_t = t_post - t_pre
        effective_rate = self.learning_rate * self.compute_modulation(t_pre, t_post)

        if delta_t > 0:
            dw = effective_rate * self.A_plus * math.exp(-delta_t / self.tau_stdp)
            self.weight += dw
        elif delta_t < 0:
            dw = effective_rate * self.A_minus * math.exp(delta_t / self.tau_stdp)
            self.weight -= dw

        self.weight = max(self.w_min, min(self.w_max, self.weight))
        return self.weight

    def record_observation(self, t_pre: float, t_post: float) -> None:
        """Record an observed pre->post delay, independent of STDP.

        Only genuinely causal pairs (``t_post > t_pre``) are recorded — this
        is tracking "what follows what," not general spike-timing
        statistics, so simultaneous or anti-causal pairs are ignored.
        Purely observational: never touches ``weight``. Callers (e.g. the
        network layer) may call this and :meth:`update_weight` for the same
        pair, but the two mechanisms are intentionally decoupled.
        """
        observed_delay = t_post - t_pre
        if observed_delay <= 0:
            return

        self.observed_delays.append(observed_delay)
        if len(self.observed_delays) > self.max_history:
            self.observed_delays.pop(0)

    @property
    def mean_observed_delay(self) -> float | None:
        """Arithmetic mean of observed delays, or None if there are none yet."""
        if not self.observed_delays:
            return None
        return sum(self.observed_delays) / len(self.observed_delays)

    @property
    def delay_variance(self) -> float | None:
        """Population variance of observed delays.

        None if fewer than 2 observations exist — variance is undefined/
        meaningless with 0 or 1 samples. Groundwork for a future confidence
        measure (low variance = high temporal regularity); this property
        exposes only the raw variance, not any derived score.
        """
        if len(self.observed_delays) < 2:
            return None
        mean = self.mean_observed_delay
        return sum((d - mean) ** 2 for d in self.observed_delays) / len(
            self.observed_delays
        )

    @property
    def confidence(self) -> float | None:
        """How consistently post follows pre at a similar delay, in [0, 1).

        confidence = regularity * sample_factor:
          - regularity = exp(-CV), where CV = sqrt(delay_variance) /
            mean_observed_delay is the *relative* dispersion (coefficient
            of variation) rather than raw variance, so a synapse with a
            long delay isn't penalized more than one with a short delay
            just for having a bigger absolute spread.
          - sample_factor = n / (n + n0): a Bayesian-shrinkage-style
            discount so a couple of coincidentally-identical observations
            can't produce high confidence on their own.

        None whenever delay_variance is None (fewer than 2 observations),
        matching V1's None-propagation convention exactly.
        """
        variance = self.delay_variance
        if variance is None:
            return None

        mean = self.mean_observed_delay
        if mean == 0.0:
            # Defensive only: V1 never records a non-positive delay, so this
            # shouldn't be reachable, but guard against a ZeroDivisionError
            # rather than let a latent bug elsewhere crash this property.
            return None

        cv = math.sqrt(variance) / mean
        regularity = math.exp(-cv)
        n = len(self.observed_delays)
        sample_factor = n / (n + self.n0)
        return regularity * sample_factor

    @property
    def future_expectation(self) -> float | None:
        """Best current guess of the pre->post delay: how long after this
        synapse's presynaptic cell fires does the postsynaptic cell
        typically fire?

        Delegates directly to mean_observed_delay (None with zero
        observations). Unlike confidence, a single observation is still a
        usable, if low-confidence, expectation — so this only requires
        n >= 1, not n >= 2.
        """
        return self.mean_observed_delay

    def compute_prediction_error(self, t_pre: float, t_post: float) -> float | None:
        """Compare an observed (t_pre, t_post) pair against prior expectation.

        Returns ``abs(actual_delay - future_expectation)`` — the raw
        mismatch between what actually happened and what this synapse
        expected beforehand. Returns None for a non-causal/simultaneous
        pair (same filter as :meth:`record_observation`), and None if
        there's no prior expectation yet (this would be the first
        observation, with nothing to compare against).

        Only handles the "actual delay mismatch" case — pre fired, post
        eventually fired, and the two delays differ. The "predicted but
        never happened" case (post never follows within a reasonable
        window) needs a timeout/scheduling mechanism that doesn't exist
        yet and is out of scope here.

        Call-order requirement: call this BEFORE :meth:`record_observation`
        for the same event. This method itself is a pure computation with
        no side effects — it never mutates state either way — but if
        ``record_observation`` runs first, ``future_expectation`` will
        already have folded in this very observation, making the
        comparison trivially shrink toward zero rather than measuring a
        genuine surprise against the prior, uncontaminated baseline.
        """
        actual_delay = t_post - t_pre
        if actual_delay <= 0:
            return None

        expectation = self.future_expectation
        if expectation is None:
            return None

        return abs(actual_delay - expectation)

    def compute_weighted_prediction_error(
        self, t_pre: float, t_post: float
    ) -> float | None:
        """``compute_prediction_error`` scaled by how much to trust it.

        The same raw mismatch is more informationally significant when it
        comes from a synapse that has established a reliable, confident
        pattern than from one still dominated by noise — a violation of a
        well-established expectation is more surprising than a wobble in
        an already-unreliable signal. Falls back to the unweighted raw
        error when ``confidence`` is None (too few samples to have a
        meaningful confidence score yet — there's nothing to weight by).
        """
        raw_error = self.compute_prediction_error(t_pre, t_post)
        if raw_error is None:
            return None

        confidence = self.confidence
        if confidence is None:
            return raw_error

        return raw_error * confidence

    def compute_modulation(self, t_pre: float, t_post: float) -> float:
        """Learning-rate multiplier derived from confidence-weighted surprise.

        modulation = m_min + (m_max - m_min) * (1 - exp(-weighted_error / tau_error))

        At weighted_error = 0 (perfectly predicted event), modulation is
        m_min (slowest learning — nothing new to learn). As weighted_error
        grows (a more surprising event, weighted by how much this synapse's
        pattern is trusted), modulation approaches m_max (fastest learning,
        bounded).

        Returns 1.0 (neutral — plain unmodulated STDP) when
        compute_weighted_prediction_error is None: either there's no prior
        expectation yet (nothing to be surprised relative to) or the pair
        is non-causal, and modulating by an undefined surprise wouldn't be
        meaningful.
        """
        weighted_error = self.compute_weighted_prediction_error(t_pre, t_post)
        if weighted_error is None:
            return 1.0

        return self.m_min + (self.m_max - self.m_min) * (
            1 - math.exp(-weighted_error / self.tau_error)
        )

    def _decay_traces(self, current_time: float) -> None:
        """Decay both STDP traces continuously up to ``current_time``.

        Same exact-exponential form as ``Cell._apply_leak``: correct
        regardless of the elapsed gap between spike events, since
        ``exp(-a) * exp(-b) == exp(-(a+b))`` makes repeated small decays
        equivalent to one large one.
        """
        elapsed = current_time - self.last_trace_update
        decay = math.exp(-elapsed / self.tau_stdp)
        self.pre_trace *= decay
        self.post_trace *= decay
        self.last_trace_update = current_time

    def on_pre_spike(self, t_pre: float, t_post_partner: float | None) -> float:
        """Apply the depression half of trace-based STDP for a presynaptic spike.

        Depresses by ``learning_rate * modulation * A_minus * post_trace``
        (evaluated *before* incrementing ``pre_trace``) — "how much recent
        post activity preceded this pre spike," the anti-causal
        contribution, equivalent to summing the ``delta_t < 0`` branch of
        :meth:`update_weight` over every past post-spike still within the
        trace's effective window.

        ``t_post_partner`` is the postsynaptic cell's most recent spike
        time (or None if it has never spiked), used only to compute
        modulation via :meth:`compute_modulation` — when None, modulation
        is neutral (1.0), since there's no partner event to be surprised
        relative to.

        A pre-spike explained by an earlier post-spike is the anti-causal
        direction, so (per the existing causal-only convention) this never
        feeds :meth:`record_observation`: the guard below only calls it
        when ``t_post_partner < t_pre``, which is exactly the ordering
        ``record_observation`` itself rejects as non-causal — so this call
        is always a documented no-op, never a live observation source. The
        forward-causal direction is handled by :meth:`on_post_spike`.
        """
        self._decay_traces(t_pre)

        modulation = (
            self.compute_modulation(t_pre, t_post_partner)
            if t_post_partner is not None
            else 1.0
        )
        dw = self.learning_rate * modulation * self.A_minus * self.post_trace
        self.weight = max(self.w_min, min(self.w_max, self.weight - dw))

        self.pre_trace += 1.0

        if t_post_partner is not None and t_post_partner < t_pre:
            self.record_observation(t_pre, t_post_partner)

        return self.weight

    def on_post_spike(self, t_post: float, t_pre_partner: float | None) -> float:
        """Apply the potentiation half of trace-based STDP for a postsynaptic spike.

        Potentiates by ``learning_rate * modulation * A_plus * pre_trace``
        (evaluated *before* incrementing ``post_trace``) — "how much recent
        pre activity preceded this post spike," the causal contribution,
        equivalent to summing the ``delta_t > 0`` branch of
        :meth:`update_weight` over every past pre-spike still within the
        trace's effective window.

        ``t_pre_partner`` is the presynaptic cell's most recent spike time
        (or None if it has never spiked); None yields neutral (1.0)
        modulation, same as :meth:`on_pre_spike`.

        This IS the forward-causal direction, so it feeds
        :meth:`record_observation` (guarded by ``t_pre_partner < t_post``,
        matching the existing causal-only convention) — mirroring how
        ``update_weight``'s call-order requirement works: modulation is
        computed from ``future_expectation`` as it stood before this
        event, and only afterward is the observation recorded.
        """
        self._decay_traces(t_post)

        modulation = (
            self.compute_modulation(t_pre_partner, t_post)
            if t_pre_partner is not None
            else 1.0
        )
        dw = self.learning_rate * modulation * self.A_plus * self.pre_trace
        self.weight = max(self.w_min, min(self.w_max, self.weight + dw))

        self.post_trace += 1.0

        if t_pre_partner is not None and t_pre_partner < t_post:
            self.record_observation(t_pre_partner, t_post)

        return self.weight
