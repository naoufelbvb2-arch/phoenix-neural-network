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

    # __slots__ eliminates the per-instance __dict__ (see the note in Cell).
    # Measured: Synapse 1,392 B -> ~232 B. SoA is still required for 1M cells.
    __slots__ = (
        "pre_id", "post_id", "weight", "distance", "propagation_speed",
        "decay_constant", "learning_rate", "tau_stdp", "A_plus", "A_minus",
        "w_min", "w_max", "observed_delays", "max_history",
        "observation_window_factor", "n0", "m_min", "m_max", "tau_error",
        "_sum_delays", "_sum_delays_sq",
        "pre_trace", "post_trace", "last_trace_update", "verify_window_factor",
        "verify_window", "n0_cs", "_pending_pre", "hits", "misses",
        "tau_decay", "decay_power", "_last_decay_time",
    )

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
        m_min: float = 0.0,
        m_max: float = 2.0,
        tau_error: float = 20.0,
        observation_window_factor: float = 3.0,
        verify_window: float | None = None,  # default: verify_window_factor * tau_stdp
        verify_window_factor: float = 0.5,
        n0_cs: float = 5.0,
        tau_decay: float = 20000.0,
        decay_power: float = 2.0,
    ) -> None:
        self.pre_id: int = pre_id
        self.post_id: int = post_id
        self.weight: float = weight
        self.distance: float = distance

        # KNOWN ARCHITECTURAL LIMIT — LONG-RANGE CONNECTIVITY IS NOT SUPPORTED.
        # With propagation_speed = 1.0, delay EQUALS distance, and
        # effective_weight = w * exp(-distance / decay_constant) makes any
        # distant synapse vanishingly weak. Measured (w capped at w_max = 20,
        # decay_constant = 10; a cell needs 25 mV from rest to fire):
        #
        #   distance   max effective weight   synapses needed to fire one cell
        #     2            16.4 mV              2
        #    10             7.4 mV              4
        #    20             2.7 mV             10
        #    30             1.0 mV             26
        #    50             0.13 mV           186
        #
        # So Phoenix currently has NO long-range shortcuts — every connection is
        # effectively local. This is a limitation of the CABLE MODEL, not of
        # verify_window: the causal horizon only refuses to learn synapses that
        # attenuation had already rendered useless.
        #
        # The correct fix, when it is needed, is FASTER AXONS, NOT SLOWER
        # SYNAPSES: raise propagation_speed for long-range projections (biology
        # myelinates for exactly this reason). A synapse with distance = 50 and
        # propagation_speed = 10 has a delay of 5 ms — comfortably inside the
        # causal horizon — while its attenuation is unchanged. Widening
        # verify_window would be the WRONG fix: it degrades noise immunity
        # without making distant synapses any stronger. DEFERRED, DELIBERATELY.
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

        # Running sums over observed_delays, maintained INCREMENTALLY in
        # record_observation so that mean_observed_delay and delay_variance are
        # O(1) reads instead of O(max_history) rescans. They were the single
        # largest remaining cost in the simulator: `confidence` -> delay_variance
        # did TWO passes over 50 elements on EVERY STDP event (826,940 calls ->
        # 37.3M inner ops, 26.9 s of a 45 s run).
        #
        # >>> NUMERICAL SAFETY — THIS DEPENDS ON THE ELIGIBILITY WINDOW <<<
        # variance = sum_d2/n - (sum_d/n)**2 is the textbook form that is
        # vulnerable to CATASTROPHIC CANCELLATION for large values with a small
        # spread. Measured: delays around 1e6 with a +/-0.001 spread give a 500x
        # wrong variance (1.22e-04 vs a true 2.5e-07).
        #
        # It is safe HERE only because the [N1] eligibility guard in
        # record_observation bounds every recorded delay to
        # observation_window_factor * tau_stdp (60 ms by default), so
        #     sum_d2 <= max_history * 60**2 = 50 * 3600 = 180,000
        # — nowhere near float64's danger zone. Verified against the naive
        # two-pass form over 300 observations: max error 3.62e-13 on variance,
        # 9.99e-16 on confidence.
        #
        # IF THAT GUARD IS EVER WIDENED OR REMOVED, re-derive this or switch to
        # Welford. Welford is numerically unconditional but O(n) — it would hand
        # the bottleneck straight back.
        self._sum_delays: float = 0.0     # sum of d  over observed_delays
        self._sum_delays_sq: float = 0.0  # sum of d^2 over observed_delays

        # Eligibility window for the OBSERVATION path: a pre/post pair further
        # apart than observation_window_factor * tau_stdp is not plausibly
        # cause-and-effect, so it is not an observation at all. Without this, a
        # post-spike after a long silence gets paired with a pre-spike from
        # hundreds of ms ago and that non-causal gap is recorded as a genuine
        # "observed delay". This restores the 3 * tau_stdp windowing that
        # predated the trace refactor — on the observation path ONLY; weights
        # keep their continuous exponential traces.
        self.observation_window_factor: float = observation_window_factor

        # V2: sample-size discount constant for `confidence`, exposed as a
        # plain mutable attribute (not baked in) — consistent with how
        # learning_rate is externally adjustable.
        self.n0: float = n0

        # V4: STDP modulation bounds. m_min/m_max are explicit tunables.
        #
        # m_min DEFAULTS TO 0.0 — the earlier tradeoff ("low m_min risks the
        # synapse freezing once it locks onto a pattern") is now RESOLVED, and
        # the resolution is the opposite of the old warning: freezing on a
        # perfect prediction is the INTENDED behavior. No error => no
        # plasticity, exactly the predictive-coding / free-energy principle
        # this project already claims. The freeze is not death: any surprise
        # instantly lifts modulation toward m_max and reawakens the synapse.
        #
        # It is also load-bearing. With m_min = 0.1 there is a permanent small
        # upward push with nothing opposing it, so the weight creeps all the
        # way to w_max and pins there (dead to learning). With m_min = 0 the
        # weight settles exactly where prediction became perfect — a
        # SELF-DETERMINED equilibrium, not a tuned constant.
        #
        # HONEST COST: with m_min = 0 the slow forgetting/drift that m_min = 0.1
        # provided is gone. A perfectly-predicted synapse stops adapting until
        # something surprises it. That is a deliberate tradeoff.
        #
        # tau_error reuses tau_stdp's default since both normalize a time-scale
        # quantity, absent a reason to differ.
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

        # Causal success tracking: P(post | pre). Every pre-spike is held here
        # awaiting verification; if the post cell fires inside verify_window it
        # is a HIT, otherwise resolve_timeouts scores it a MISS. See the
        # causal_success property for why this is NOT redundant with confidence.
        #
        # verify_window is DERIVED from tau_stdp, not hardcoded. It defines the
        # network's CAUSAL HORIZON, and it must be consistent with the window
        # over which weights actually learn: a hardcoded 6 ms against a 20 ms
        # tau_stdp meant an 8 ms partner (an ordinary biological latency) was
        # scored a full MISS and depressed as if it were noise, while STDP
        # itself was happily learning from it. The two mechanisms disagreed
        # about what "causal" means.
        #
        # verify_window is a HARD cutoff (binary yes/no); tau_stdp is a SOFT
        # exponential influence range. They are different kinds of quantity and
        # need not be equal — but the cutoff must lie INSIDE the influence range
        # and be justified from it. Hence a factor, and hence 0.5.
        #
        # MEASURED TRADEOFF (weight gap = w(loop) - w(noise); positive = the
        # network can still tell a causal loop from noise):
        #
        #   k     window   noise 5%   noise 10%   noise 20%
        #   0.5   10 ms    +1.52      +5.74       +10.06     <- chosen
        #   0.75  15 ms    +1.20      +4.07        +3.24  (a seed failed)
        #   1.0   20 ms    +0.98      +2.63        +0.52  (collapsed)
        #
        # A wider window recognises slower causation but gives a noise synapse
        # more chances to harvest free hits from a regularly-firing post cell,
        # and the discrimination collapses. k = 0.5 sees 8 ms causation while
        # keeping a large noise margin.
        self.verify_window_factor: float = verify_window_factor
        self.verify_window: float = (
            verify_window
            if verify_window is not None
            else verify_window_factor * self.tau_stdp
        )
        self.n0_cs: float = n0_cs
        self._pending_pre: list[float] = []
        self.hits: int = 0
        self.misses: int = 0

        # Passive weight decay, gated by causal reliability. See apply_decay.
        self.tau_decay: float = tau_decay
        self.decay_power: float = decay_power
        self._last_decay_time: float = 0.0

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
        """REFERENCE IMPLEMENTATION ONLY — not used by any live simulation path.

        Retained to document the equivalence between this pairwise form and the
        trace-based :meth:`on_pre_spike` / :meth:`on_post_spike`, which are what
        every live path (``Network``, ``TwoCellNetwork``) actually calls. **Do
        not call from network code**: two live implementations of the same rule
        can silently diverge. Its own tests are the equivalence documentation.

        Applies one prediction-error-modulated STDP update.

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

        DECLARED MODEL: **the synapse predicts the delay of the TRIGGERING
        spike** — the presynaptic spike most immediately preceding the
        postsynaptic one (callers pass the partner's ``last_spike_time``).
        This is a deliberate, tested modeling choice, not an implementation
        artifact. All-pairs semantics (recording EVERY recent pre-spike
        against the post-spike) was evaluated and **rejected**: it cannot
        converge. With pre-spikes at offsets {10, 2} before the post-spike it
        yields ``mean_observed_delay = 6`` ms — an arithmetic midpoint
        matching no real physical delay — so prediction_error stays pinned at
        ~4 ms forever, and it reports ``confidence ~0.47`` on a ZERO-noise
        system by conflating *structural spread* (a fixed fact of the wiring)
        with *timing jitter* (the irregularity confidence is meant to
        measure). See ``test_all_pairs_semantics_cannot_converge``.

        Weights and prediction are asymmetric FOR A GOOD REASON — this is not
        an inconsistency to "fix": weights SUM contributions, so all-pairs
        traces are correct there (summation over heterogeneous events is
        legitimate); prediction ESTIMATES A DISTRIBUTION, which requires a
        homogeneous sample, and mixing two structurally distinct delays into
        one distribution makes convergence mathematically impossible.

        Two guards, both required:
        - Only genuinely causal pairs (``t_post > t_pre``) — this tracks "what
          follows what", so simultaneous/anti-causal pairs are ignored.
        - Only pairs inside the eligibility window
          (``observation_window_factor * tau_stdp``) — a pair further apart
          than that is not plausibly cause-and-effect, and recording it would
          poison the distribution with a non-causal gap.

        Purely observational: never touches ``weight``. Callers may call this
        and :meth:`update_weight` for the same pair; the two are decoupled.
        """
        observed_delay = t_post - t_pre
        if observed_delay <= 0:
            return
        if observed_delay > self.observation_window_factor * self.tau_stdp:
            return  # too far apart to be a causal source; not an observation

        # Maintain the running sums around the FIFO eviction. observed_delays
        # itself keeps exactly its old type, contents and FIFO semantics.
        if len(self.observed_delays) == self.max_history:
            evicted = self.observed_delays[0]
            self._sum_delays -= evicted
            self._sum_delays_sq -= evicted * evicted

        self.observed_delays.append(observed_delay)
        if len(self.observed_delays) > self.max_history:
            self.observed_delays.pop(0)

        self._sum_delays += observed_delay
        self._sum_delays_sq += observed_delay * observed_delay

    def _recompute_sums(self) -> None:
        """TEST-SUPPORT ESCAPE HATCH — not API. Do not call from live code.

        The running sums are INTERNAL STATE owned by record_observation; direct
        mutation of ``observed_delays`` is not a supported operation and leaves
        them stale. Exactly one test needs it: the defensive zero-mean guard in
        ``confidence`` covers a state that is UNREACHABLE through the public API
        (record_observation rejects any delay <= 0, so the mean can never be 0),
        and so must be constructed by assigning observed_delays directly.
        """
        self._sum_delays = sum(self.observed_delays)
        self._sum_delays_sq = sum(d * d for d in self.observed_delays)

    @property
    def mean_observed_delay(self) -> float | None:
        """Arithmetic mean of observed delays, or None if there are none yet.

        O(1): read from the running sum, not a rescan. See _sum_delays.
        """
        n = len(self.observed_delays)
        if n == 0:
            return None
        return self._sum_delays / n

    @property
    def delay_variance(self) -> float | None:
        """Population variance of observed delays.

        None if fewer than 2 observations exist — variance is undefined/
        meaningless with 0 or 1 samples. Groundwork for a future confidence
        measure (low variance = high temporal regularity); this property
        exposes only the raw variance, not any derived score.

        O(1): computed from the running sums as sum_d2/n - (sum_d/n)**2 rather
        than a two-pass rescan. See the numerical-safety note on _sum_delays —
        this form is only safe because the eligibility window bounds the delays.
        """
        n = len(self.observed_delays)
        if n < 2:
            return None

        variance = self._sum_delays_sq / n - (self._sum_delays / n) ** 2

        # REQUIRED clamp. Rounding can push a true variance of exactly 0 a hair
        # below zero — and an all-identical delay sequence (a perfectly predicting
        # loop) is our most common steady state. A negative here would make the
        # sqrt() in `confidence` blow up.
        return max(0.0, variance)

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

        KNOWN LIMIT (not fixed): dt quantization INFLATES this score. Observed
        delays are whole multiples of the simulation ``dt``, so any jitter
        finer than one tick is invisible — variance is under-reported and
        regularity therefore over-reported. A synapse can look perfectly
        regular simply because its jitter is smaller than the clock's
        resolution.
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

    @property
    def causal_success(self) -> float | None:
        """P(post | pre) — of all MY spikes, how many actually worked?

        ``confidence`` asks "when it worked, how regular was it?" — it is
        CONDITIONED ON SUCCESS and is therefore blind to false positives. It
        never counts the times this synapse fired and nothing happened. That is
        confirmation bias encoded in math: if the post cell fires regularly for
        its own reasons, ANY spike arriving just before it looks perfectly
        causal (``observed_delays = [1.0, 1.0, ...]``, ``variance = 0``). The
        rooster crows every morning and concludes it causes the sunrise.

        ``causal_success`` asks the other question — "of all my spikes, how many
        actually worked?" — by counting the MISSES. A noise synapse fires
        constantly and mostly nothing follows, so it is punished by its own
        failures. Measured against a regularly-firing post cell: a noise synapse
        scores ``confidence`` ~0.909 (indistinguishable from a true causal loop,
        which also scores ~0.909) but ``causal_success`` ~0.19.

        **Both are needed. Neither subsumes the other:** ``confidence`` measures
        timing REGULARITY, ``causal_success`` measures causal RELIABILITY.

        Discounted by ``n0_cs`` in the same Bayesian-shrinkage spirit as
        ``confidence``'s ``n0``, so a couple of lucky hits cannot mint a high
        score. None until there is any evidence at all.

        THE CAUSAL HORIZON. ``verify_window`` (default ``0.5 * tau_stdp`` =
        10 ms) is what this measurement is relative to: a presynaptic partner
        whose spike is not followed by a post-spike inside that window is scored
        a MISS and judged non-causal. That is a deliberate modeling decision,
        not an artifact.

        **It limits SYNAPTIC LATENCY, not SEQUENCE LENGTH.** This distinction is
        the difference between a harmless constraint and a crippling one. A long
        temporal sequence is learned as a CHAIN of short causal links — verified:
        a 4-cell chain (1->2->3->4) with 3 ms hops learns end-to-end, every
        synapse reaching ``causal_success`` 0.889, because each synapse only ever
        sees its own immediate 3 ms neighbour. A 500 ms pattern can be learned by
        150 synapses each seeing 3 ms. What the horizon forbids is a SINGLE
        synapse jumping a gap longer than the window (a direct 25 ms-latency
        synapse scores 0.0 and is pruned) — and cable attenuation had ALREADY
        made such a synapse useless (see the long-range note at
        ``decay_constant``), so the cutoff and the cable physics agree rather
        than conflict.

        >>> THE CAUSAL DISCRIMINATION LAW <<<

            post_firing_rate * verify_window  <<  1        (i.e. ISI >> verify_window)

        If the post cell fires in nearly EVERY verification window, then a noise
        spike and a causal spike produce IDENTICAL observations. The causal
        information is NOT PRESENT IN THE DATA, and no local, observational
        measure can recover it — this is an information-theoretic limit, not a
        tuning issue. (A baseline-lift measure, P(post|pre) - P(post), was tried:
        it returns exactly 0.0 for everything, because P(post) = 1.0. The measure
        did not fail; it correctly reported that no information exists.)

        Measured, with verify_window = 10 ms:

            post rate   ISI     P(post in a random window)   discrimination
              5 Hz      200ms          0.05                  works
             20 Hz       50ms          0.20                  works
             33 Hz       30ms          0.33                  works
             50 Hz       20ms          0.50                  marginal
            100 Hz       10ms          1.00                  IMPOSSIBLE
            250 Hz        4ms          1.00                  IMPOSSIBLE

        And the consequence, measured on a 10-cell ring:

            hop   post rate   cs(loop)   cs(noise)   w(noise)
             1     100 Hz      0.998      0.902      20.0 = w_max   <- CATASTROPHE
             3      33 Hz      0.998      0.320       6.88          <- pruned
             5      20 Hz      0.996      0.161       7.19          <- pruned

        At 100 Hz the NOISE synapse out-competes the real assembly and pins at
        w_max. This is the rooster problem returning at the network level: a cell
        firing every 10 ms makes ANY incoming spike look causal.

        CONSEQUENCE: the network MUST operate in a sparse-firing regime (<~50 Hz
        with a 10 ms window). This is not a preference — it is a PRECONDITION for
        causal learning to be possible at all. It is very likely why biological
        networks fire at 1-20 Hz: sparse coding is not a metabolic luxury but an
        information-theoretic requirement.

        THE RATE IS SET BY TOPOLOGY (the cycle period), NOT BY HOMEOSTASIS.
        Homeostasis was tested and CANNOT enforce this Law: reverberation is
        BISTABLE (all-or-nothing), so raising Vthresh does not slow the loop, it
        EXTINGUISHES it (measured: 100 Hz -> 0 Hz within 2 s, after which Vthresh
        sank to its floor with nothing left to re-ignite). Homeostasis assumes a
        graded system; reverberation is not one.

        KNOWN LIMIT: this is P(post|pre) — still CORRELATIONAL, not true
        causality. It does not ask the counterfactual ("would post have fired
        WITHOUT pre?"). It holds up to ~20% noise in our tests, but a very
        densely-firing post cell can still let a noise synapse harvest free hits
        (which is exactly what the Law above quantifies).
        """
        n = self.hits + self.misses
        if n == 0:
            return None
        return (self.hits / n) * (n / (n + self.n0_cs))

    def resolve_timeouts(self, now: float) -> None:
        """Score any pre-spike whose verification window has PASSED as a MISS.

        This is the ONLY place misses are counted.

        >>> IT IS CALLED AT EVENT BOUNDARIES, NOT EVERY TICK <<<
        Every method that touches this synapse's pending state (``on_pre_spike``,
        ``on_post_spike``, ``apply_decay``) settles stale pendings by calling this
        FIRST. That keeps the cost model at O(spikes) instead of
        O(synapses x ticks) — a per-tick sweep was the single worst cost in the
        simulator (1.2M calls per 1000 ticks on 1200 synapses).

        DO NOT "optimize" this by simply DELETING the call. That silently breaks
        the system: a synapse whose pre fires repeatedly with no post response
        would accumulate _pending_pre without bound (a memory leak, precisely in
        the noisy synapse we most need to prune) while reporting misses = 0 and
        causal_success = None — i.e. "no evidence" when it has in fact FAILED
        every time. Relocating the call preserves the accounting exactly;
        removing it destroys it.

        BOUNDARY CONVENTION: a post-spike at exactly ``t_pre + verify_window`` is
        a HIT. This method must therefore use a STRICT ``>``. Using ``>=`` was a
        bug: ``Network.step`` calls resolve_timeouts (step d2) BEFORE
        on_post_spike (step e), so a ``>=`` here consumed the pending spike as a
        miss before on_post_spike — whose own guard is
        ``t_pre < t_post <= t_pre + verify_window`` — ever saw it. That made
        on_post_spike's ``<=`` branch DEAD CODE, and the two guards stated
        opposite intentions about the same instant, with the earlier one silently
        winning.

        The damage was real: a PERFECTLY CAUSAL synapse whose latency landed
        exactly on the horizon scored 0 hits / 50 misses -> causal_success = 0.0,
        and was decayed away as pure noise despite its post cell following it
        every single time. A pre-spike is a miss only once the window has been
        PASSED, never when it is merely REACHED.
        """
        still_pending: list[float] = []
        for t_pre in self._pending_pre:
            if now > t_pre + self.verify_window:
                self.misses += 1  # fired, and nothing followed
            else:
                still_pending.append(t_pre)
        self._pending_pre = still_pending

    def apply_decay(self, now: float) -> None:
        """Lazy, event-driven weight decay gated by causal reliability.

        Applied ON DEMAND (not every tick): the accumulated decay over the whole
        elapsed interval is computed in ONE exponential step. This keeps the
        simulation O(spikes) rather than O(synapses x ticks) — essential to the
        neuromorphic event-driven cost model, and therefore a correctness
        requirement of the design, not a micro-optimization.

        Gating: ``leak ∝ (1 - causal_success) ** decay_power``.

        - An IDLE synapse (never fired: hits = misses = 0 -> causal_success is
          None -> treated as 0.0) decays at the FULL rate and is pruned. This is
          the ONLY mechanism in the system that prunes a synapse which never
          fires: causal_success alone cannot, since with no events it has no
          opinion and the weight would otherwise sit at its initial value
          forever, contributing dead charge to fan-in ([OPEN-1]).

        - A PROVEN CAUSAL synapse is protected. Quadratic gating is load-bearing
          here, not cosmetic:

          * BLIND decay (no gating) is WRONG — measured: it drags everything down
            equally (loop 7.78 vs noise 7.83, i.e. ZERO discrimination) and kills
            the loop outright.

          * LINEAR gating (decay_power = 1) is TOO WEAK. A real loop's
            causal_success saturates at ~0.996, never 1.0, because of the
            Bayesian discount n/(n + n0_cs). That residual leaves a permanent
            ~0.4% bleed which killed the loop depending on the RNG seed.

          * QUADRATIC gating (decay_power = 2) fixes it: over 100 s a causal
            synapse (cs = 0.996) loses 0.008% instead of 2.0% — ~250x less —
            while an idle synapse (cs = 0) decays at the full, ungated rate.
        """
        # Settle stale pendings FIRST: the leak is GATED by causal_success, so a
        # stale cs would mis-gate the decay. apply_decay already runs lazily (on
        # the post cell's firing), so this keeps cs fresh exactly where it is used.
        self.resolve_timeouts(now)

        elapsed = now - self._last_decay_time
        if elapsed <= 0:
            return

        causal_success = self.causal_success
        if causal_success is None:
            causal_success = 0.0  # no evidence of ever having worked -> not spared

        leak = ((1.0 - causal_success) ** self.decay_power) / self.tau_decay
        self.weight = max(self.w_min, self.weight * math.exp(-leak * elapsed))
        self._last_decay_time = now

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
        direction, so (per the causal-only convention) this never feeds
        :meth:`record_observation`. The forward-causal direction is handled
        by :meth:`on_post_spike`.
        """
        # Settle stale pendings FIRST (event-boundary accounting, not per-tick).
        self.resolve_timeouts(t_pre)

        self._decay_traces(t_pre)

        # This spike is now awaiting verification: did the post cell actually
        # follow? on_post_spike scores a hit; resolve_timeouts scores a miss.
        self._pending_pre.append(t_pre)

        modulation = (
            self.compute_modulation(t_pre, t_post_partner)
            if t_post_partner is not None
            else 1.0
        )
        dw = self.learning_rate * modulation * self.A_minus * self.post_trace

        # Gate DEPRESSION by unreliability: a synapse that reliably causes its
        # post cell is PROTECTED from the loop's self-inflicted anti-causal
        # depression (in a cycle every synapse is anti-causal to itself on the
        # next lap). An unreliable one takes the full hit. Neutral (unscaled)
        # while there is no evidence yet.
        causal_success = self.causal_success
        if causal_success is not None:
            dw *= 1.0 - causal_success

        self.weight = max(self.w_min, min(self.w_max, self.weight - dw))

        self.pre_trace += 1.0

        # No record_observation() here by construction: the anti-causal
        # direction (t_post_partner < t_pre) is exactly what record_observation
        # rejects, so calling it could only ever be a no-op.

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
        # Settle stale pendings FIRST. A pending whose window has been PASSED is
        # a miss and must not be creditable here; one landing exactly ON the
        # horizon survives (strict `>`), and is credited as a HIT just below.
        self.resolve_timeouts(t_post)

        self._decay_traces(t_post)

        # CONFIRM pending pre-spikes: any that fired inside the verification
        # window before this post-spike actually worked -> HIT. (Those left
        # pending are still awaiting their verdict; resolve_timeouts will score
        # them a miss once their window expires.)
        still_pending: list[float] = []
        for t_pending in self._pending_pre:
            if t_pending < t_post <= t_pending + self.verify_window:
                self.hits += 1
            else:
                still_pending.append(t_pending)
        self._pending_pre = still_pending

        modulation = (
            self.compute_modulation(t_pre_partner, t_post)
            if t_pre_partner is not None
            else 1.0
        )
        dw = self.learning_rate * modulation * self.A_plus * self.pre_trace

        # Gate POTENTIATION by reliability: an unreliable synapse (one whose
        # spikes mostly are NOT followed by the post cell) barely potentiates,
        # however regular its lucky hits happen to look. Neutral (unscaled)
        # while there is no evidence yet.
        causal_success = self.causal_success
        if causal_success is not None:
            dw *= causal_success

        self.weight = max(self.w_min, min(self.w_max, self.weight + dw))

        self.post_trace += 1.0

        if t_pre_partner is not None and t_pre_partner < t_post:
            self.record_observation(t_pre_partner, t_post)

        return self.weight
