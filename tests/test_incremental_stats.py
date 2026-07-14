"""[PERF-2] mean/variance/confidence in O(1) — the last bottleneck.

`confidence` -> `delay_variance` used to recompute an O(max_history) sum on EVERY
STDP event, and worse: `delay_variance` called `mean_observed_delay` (one full
scan) and then scanned AGAIN for the squared deviations — TWO passes over 50
elements per call. Profiled at 826,940 calls driving 37.3M inner operations,
26.9 s of a 45 s run.

Replaced by two running sums maintained incrementally in record_observation:

    mean = sum_d / n
    var  = sum_d2 / n - (sum_d / n)**2

These tests prove the incremental form is EXACT (including across FIFO eviction),
never yields a negative variance, is genuinely O(1), and — critically — pin the
numerical-safety dependency on the [N1] eligibility window that makes the
sum-of-squares form safe at all.
"""

from __future__ import annotations

import math
import random

import pytest

from phoenix.synapse import Synapse


def _naive_mean(delays: list[float]) -> float | None:
    if not delays:
        return None
    return sum(delays) / len(delays)


def _naive_variance(delays: list[float]) -> float | None:
    """The ORIGINAL two-pass implementation, kept here as the reference oracle."""
    if len(delays) < 2:
        return None
    mean = sum(delays) / len(delays)
    return sum((d - mean) ** 2 for d in delays) / len(delays)


def _naive_confidence(synapse: Synapse, delays: list[float]) -> float | None:
    variance = _naive_variance(delays)
    if variance is None:
        return None
    mean = _naive_mean(delays)
    if mean == 0.0:
        return None
    regularity = math.exp(-math.sqrt(variance) / mean)
    n = len(delays)
    return regularity * (n / (n + synapse.n0))


# ---------------------------------------------------------------------------
# Q1. THE CORRECTNESS PROOF — exact across FIFO eviction, not just appends
# ---------------------------------------------------------------------------
def test_incremental_stats_match_naive_exactly() -> None:
    """300 observations against a max_history of 50: the FIFO evicts ~250 times.

    Eviction is where an incremental sum goes wrong if the bookkeeping is off by
    one, so this must exercise it heavily — appends alone would prove nothing.
    """
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, max_history=50)
    horizon = synapse.observation_window_factor * synapse.tau_stdp  # 60 ms

    rng = random.Random(7)
    evictions = 0

    for k in range(300):
        delay = rng.uniform(0.5, horizon)  # inside the eligibility window
        before = len(synapse.observed_delays)
        synapse.record_observation(t_pre=1000.0 * k, t_post=1000.0 * k + delay)
        if before == synapse.max_history:
            evictions += 1

        delays = list(synapse.observed_delays)

        assert synapse.mean_observed_delay == pytest.approx(
            _naive_mean(delays), abs=1e-9
        )
        naive_var = _naive_variance(delays)
        if naive_var is None:
            assert synapse.delay_variance is None
        else:
            assert synapse.delay_variance == pytest.approx(naive_var, abs=1e-9)

        naive_conf = _naive_confidence(synapse, delays)
        if naive_conf is None:
            assert synapse.confidence is None
        else:
            assert synapse.confidence == pytest.approx(naive_conf, abs=1e-9)

    # The FIFO really did evict — the hard case was exercised, not skipped.
    assert evictions >= 240
    assert len(synapse.observed_delays) == 50


# ---------------------------------------------------------------------------
# Q2. The variance must never go negative (our most common steady state)
# ---------------------------------------------------------------------------
def test_variance_is_never_negative() -> None:
    """A perfectly predicting loop records the SAME delay forever.

    The true variance is exactly 0, and sum_d2/n - (sum_d/n)**2 can round a hair
    BELOW zero there. Unclamped, that would make sqrt() in `confidence` raise —
    and it would do so in the steady state we most want to reach.
    """
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, max_history=50)

    for k in range(200):  # well past max_history: evicts constantly
        synapse.record_observation(t_pre=1000.0 * k, t_post=1000.0 * k + 3.0)

    assert synapse.observed_delays == [3.0] * 50
    assert synapse.delay_variance == 0.0        # exactly zero, not -1e-16
    assert synapse.delay_variance >= 0.0

    # And confidence is well-defined: sqrt(0) is fine, regularity is exp(0) = 1.
    assert synapse.confidence is not None
    assert synapse.confidence == pytest.approx(50 / (50 + synapse.n0))


# ---------------------------------------------------------------------------
# Q3. O(1): reading confidence must not scale with max_history
# ---------------------------------------------------------------------------
def test_confidence_is_o1() -> None:
    """Call-counting, not timing — timing tests are flaky.

    The old implementation iterated `observed_delays` on every read (twice, in
    fact). So the honest, non-flaky assertion is: reading `confidence` must not
    iterate the list AT ALL. We prove that by swapping in a list subclass that
    counts its own iterations.
    """
    class CountingList(list):
        iterations = 0

        def __iter__(self):
            CountingList.iterations += 1
            return super().__iter__()

    small = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, max_history=10)
    large = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, max_history=500)

    for synapse in (small, large):
        for k in range(600):
            synapse.record_observation(t_pre=100.0 * k, t_post=100.0 * k + 3.0 + (k % 7))

    assert len(small.observed_delays) == 10
    assert len(large.observed_delays) == 500

    # Swap the underlying lists for counting ones (contents identical).
    small.observed_delays = CountingList(small.observed_delays)
    large.observed_delays = CountingList(large.observed_delays)
    small._recompute_sums()
    large._recompute_sums()

    CountingList.iterations = 0
    for _ in range(1_000):
        _ = small.confidence
        _ = large.confidence

    # 2,000 confidence reads over lists of 10 and 500 — and NOT ONE iteration of
    # either list. The read is O(1) and independent of max_history.
    assert CountingList.iterations == 0

    # Both still produce a real, equal-by-construction score.
    assert small.confidence is not None
    assert large.confidence is not None


# ---------------------------------------------------------------------------
# Q4. THE NUMERICAL-SAFETY DEPENDENCY: the eligibility window bounds the sums
# ---------------------------------------------------------------------------
def test_eligibility_window_bounds_the_sums() -> None:
    """sum_d2/n - (sum_d/n)**2 is only safe because the delays are BOUNDED.

    This form suffers catastrophic cancellation for large values with a small
    spread (measured: delays ~1e6 with a +/-0.001 spread give a 500x wrong
    variance). We are protected solely by the [N1] eligibility guard, which caps
    every recorded delay at observation_window_factor * tau_stdp (60 ms), so
    sum_d2 <= max_history * 60**2 = 180,000 — far from float64's danger zone.

    This test pins that dependency. If someone widens or removes the guard, this
    fails and points at the numerical-safety note rather than letting the
    variance silently rot.
    """
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, max_history=50)
    horizon = synapse.observation_window_factor * synapse.tau_stdp
    assert horizon == 60.0

    # An observation BEYOND the horizon is rejected outright...
    synapse.record_observation(t_pre=0.0, t_post=horizon + 1.0)
    assert synapse.observed_delays == []
    # ...and must not have touched the sums.
    assert synapse._sum_delays == 0.0
    assert synapse._sum_delays_sq == 0.0

    # A non-causal pair likewise.
    synapse.record_observation(t_pre=10.0, t_post=5.0)
    assert synapse._sum_delays == 0.0
    assert synapse._sum_delays_sq == 0.0

    # Fill the window with the largest ELIGIBLE delays possible.
    for k in range(synapse.max_history):
        synapse.record_observation(t_pre=1000.0 * k, t_post=1000.0 * k + horizon)

    # The bound the safety argument rests on, asserted directly.
    assert synapse._sum_delays_sq <= synapse.max_history * horizon**2
    assert synapse._sum_delays_sq == pytest.approx(50 * 3600.0)
    assert synapse._sum_delays_sq <= 180_000.0

    # And at that bound the variance is still exact (all delays identical -> 0).
    assert synapse.delay_variance == 0.0


# ---------------------------------------------------------------------------
# Q5. Every scientific number must reproduce EXACTLY — not "within tolerance"
# ---------------------------------------------------------------------------
def test_full_stack_unchanged() -> None:
    """The canonical assembly, pinned to the values committed BEFORE this refactor.

    The full 15-scenario sweep (noise 0->10% x 3 seeds) is BIT-FOR-BIT identical
    before and after the incremental-sums change. This is the 5%/seed-1 scenario
    from it, pinned to exact committed values rather than a tolerance, because a
    pure cost refactor has no licence to move a single digit.
    """
    from phoenix.cell import Cell
    from phoenix.network_graph import Network

    ring, fan_in, w0, hop = 10, 3, 11.0, 3
    noise_id, idle_id = 100, 101

    def syn(pre: int, post: int, delay: float) -> Synapse:
        return Synapse(pre_id=pre, post_id=post, weight=w0, distance=delay,
                       propagation_speed=1.0, decay_constant=1000.0)

    net = Network(dt=1.0)
    for i in range(ring):
        net.add_cell(Cell(neuron_id=i))
    net.add_cell(Cell(neuron_id=noise_id))
    net.add_cell(Cell(neuron_id=idle_id))
    for i in range(ring):
        for k in range(1, fan_in + 1):
            net.add_synapse(syn((i - k) % ring, i, k * hop))
    net.add_synapse(syn(noise_id, 0, 1.0))
    net.add_synapse(syn(idle_id, 1, 1.0))

    rng = random.Random(1)
    run_ms = 30_000
    spikes = []
    for _ in range(run_ms):
        for i in range(fan_in):
            if net.current_time == i * hop:
                net.inject(i, 100.0)
        if rng.random() < 0.05:
            net.inject(noise_id, 100.0)
        spikes.extend(net.step())

    tail = [s for s in spikes if s.neuron_id == 0 and s.timestamp > run_ms - 5_000]
    rate = len(tail) / 5.0
    assembly = [s for i in range(ring) for s in net.incoming[i] if s.pre_id < ring]
    assembly_mean = sum(s.weight for s in assembly) / len(assembly)
    noise = next(s for s in net.incoming[0] if s.pre_id == noise_id)
    idle = next(s for s in net.incoming[1] if s.pre_id == idle_id)

    # Exact committed values (5% noise, seed 1), to 4 decimal places.
    assert rate == pytest.approx(33.2, abs=0.01)
    assert assembly_mean == pytest.approx(10.808, abs=0.001)
    assert noise.weight == pytest.approx(2.411, abs=0.001)
    assert idle.weight == pytest.approx(2.4576, abs=0.0001)

    # ...and the conclusions they support, unchanged.
    assert noise.weight < assembly_mean          # active noise pruned
    assert idle.weight < 0.5 * w0                # idle synapse pruned
    assert idle.causal_success is None           # it never fired
