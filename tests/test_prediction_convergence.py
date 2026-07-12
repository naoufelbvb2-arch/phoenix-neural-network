"""The project's central claim: learning quiets down as prediction proves correct.

Nothing in the suite previously tested this end-to-end. These tests close it:

    a fully PREDICTED pattern must stop driving learning (modulation -> m_min),
    and a SURPRISING event must start driving it again (modulation -> m_max).

They also pin the settled observation semantics ([N1-a]) so it cannot silently
regress: the synapse predicts the delay of the TRIGGERING spike (the presynaptic
spike most immediately preceding the postsynaptic one). All-pairs semantics was
evaluated and REJECTED because it cannot converge — that rejection is itself
encoded below as a documentation-test.
"""

from __future__ import annotations

from phoenix.synapse import Synapse

TAU_STDP = 20.0
DELAY = 10.0        # the one true physical delay of the perfect pattern
SPACING = 300.0     # trials spaced >> tau_stdp, so nothing bleeds across trials
N_TRIALS = 30


def _synapse(**kwargs: float) -> Synapse:
    return Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=1.0,
        tau_stdp=TAU_STDP, m_min=0.1, m_max=2.0, tau_error=20.0,
        **kwargs,
    )


def _train_perfect_pattern(synapse: Synapse, n_trials: int = N_TRIALS) -> None:
    """Pre at `base`, post at `base + DELAY`, repeated — zero physical noise.

    Call order matters and mirrors the live path: the modulation for an event is
    computed against the expectation as it stood BEFORE that event, so the
    observation is only recorded afterwards.
    """
    for k in range(n_trials):
        base = 100.0 + k * SPACING
        synapse.record_observation(t_pre=base, t_post=base + DELAY)


# ---------------------------------------------------------------------------
# THE central claim: predicted => quiet
# ---------------------------------------------------------------------------
def test_prediction_error_converges_to_zero_on_perfect_pattern() -> None:
    synapse = _synapse()
    _train_perfect_pattern(synapse)

    # A late trial, identical in form to every earlier one.
    base = 100.0 + N_TRIALS * SPACING
    error = synapse.compute_prediction_error(t_pre=base, t_post=base + DELAY)
    modulation = synapse.compute_modulation(t_pre=base, t_post=base + DELAY)

    # The synapse has learned the pattern exactly: nothing left to be surprised by.
    assert error is not None
    assert abs(error) < 1e-9

    # ...so learning quiets all the way down to its floor. This is the claim.
    assert abs(modulation - synapse.m_min) < 1e-9

    # And it got there honestly: a single, homogeneous delay, zero variance.
    assert synapse.future_expectation == DELAY
    assert synapse.delay_variance == 0.0


# ---------------------------------------------------------------------------
# The other half of the loop: surprising => learn again
# ---------------------------------------------------------------------------
def test_surprise_raises_modulation() -> None:
    synapse = _synapse()
    _train_perfect_pattern(synapse)

    base = 100.0 + N_TRIALS * SPACING
    quiet = synapse.compute_modulation(t_pre=base, t_post=base + DELAY)

    # ONE event with a wildly different delay (5x the learned one).
    surprising = synapse.compute_modulation(t_pre=base, t_post=base + 5 * DELAY)

    # Learning jumps back on: well above the floor, and heading for the ceiling.
    assert surprising > quiet
    assert surprising > synapse.m_min
    assert surprising > 0.5 * synapse.m_max

    # Predicted -> quiet, surprising -> learn. That is the loop, closed.
    assert abs(quiet - synapse.m_min) < 1e-9


# ---------------------------------------------------------------------------
# [N1-a] the eligibility window
# ---------------------------------------------------------------------------
def test_stale_pair_beyond_window_is_not_recorded() -> None:
    synapse = _synapse()  # default factor 3.0 -> window = 3 * 20 = 60 ms
    window = synapse.observation_window_factor * synapse.tau_stdp
    assert window == 60.0

    # Just INSIDE the window: a plausible cause -> recorded.
    synapse.record_observation(t_pre=0.0, t_post=window - 1.0)
    assert synapse.observed_delays == [59.0]

    # Beyond the window: a post-spike after a long silence, paired with a
    # pre-spike from ages ago. Not causally related -> NOT an observation.
    synapse.record_observation(t_pre=0.0, t_post=window + 1.0)
    synapse.record_observation(t_pre=0.0, t_post=500.0)
    assert synapse.observed_delays == [59.0]  # unchanged

    # Exactly ON the boundary is still eligible (the guard is strict `>`).
    synapse.record_observation(t_pre=0.0, t_post=window)
    assert synapse.observed_delays == [59.0, 60.0]


# ---------------------------------------------------------------------------
# [N1-a] documentation-as-test: WHY all-pairs was rejected
# ---------------------------------------------------------------------------
def test_all_pairs_semantics_cannot_converge() -> None:
    """This documents WHY all-pairs was rejected. It is NOT the implemented behavior.

    Under all-pairs, a post-spike is paired with EVERY recent pre-spike, not just
    the triggering one. Here the pre-spikes sit 10 ms and 2 ms before the post,
    so all-pairs records BOTH delays every trial. The resulting expectation is
    their arithmetic midpoint (6 ms) — a delay that matches NO real physical
    event — so prediction_error can never reach zero, no matter how perfectly
    regular and noise-free the pattern is.

    It also fabricates uncertainty: on a system with ZERO physical jitter it
    reports a large variance, because it conflates STRUCTURAL spread (two spikes
    at two fixed offsets — a fact about the wiring) with TIMING jitter (real
    irregularity, which is what confidence is supposed to measure).
    """
    synapse = _synapse()

    for k in range(N_TRIALS):
        base = 100.0 + k * SPACING
        t_post = base + 10.0
        # All-pairs: record the post against BOTH preceding pre-spikes.
        synapse.record_observation(t_pre=base, t_post=t_post)         # delay 10
        synapse.record_observation(t_pre=base + 8.0, t_post=t_post)   # delay  2

    # The expectation is a midpoint matching neither real delay.
    assert synapse.future_expectation == 6.0

    # So the error is permanently pinned at 4 ms — it does NOT converge to zero,
    # on a perfectly regular, entirely noise-free pattern.
    base = 100.0 + N_TRIALS * SPACING
    error = synapse.compute_prediction_error(t_pre=base, t_post=base + 10.0)
    assert error == 4.0
    assert error > 1e-9  # never converges — the reason all-pairs is rejected

    # And it invents jitter that does not exist: variance is large and confidence
    # is middling on a ZERO-noise system.
    assert synapse.delay_variance == 16.0
    assert synapse.confidence is not None
    assert 0.4 < synapse.confidence < 0.55

    # Contrast: the IMPLEMENTED (triggering-spike) semantics on the same pattern
    # converges to exactly zero.
    implemented = _synapse()
    _train_perfect_pattern(implemented)
    implemented_error = implemented.compute_prediction_error(
        t_pre=base, t_post=base + DELAY
    )
    assert abs(implemented_error) < 1e-9
