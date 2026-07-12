"""The causal horizon: verify_window = 0.5 * tau_stdp.

CST shipped with a hardcoded ``verify_window = 6.0`` ms against a ``tau_stdp`` of
20 ms. The two mechanisms therefore disagreed about what "causal" means: STDP
happily learned from a partner firing 8 ms before the post cell, while
causal_success scored that same partner a full MISS and depressed it as if it
were noise. 6.0 was an arbitrary number from a prototype tuned to a 4 ms loop.

The window is now DERIVED: ``verify_window = verify_window_factor * tau_stdp``,
factor 0.5 -> 10 ms. It is a HARD cutoff living inside a SOFT exponential
influence range, which is why it is a fraction of tau_stdp rather than equal to it.

THE KEY POINT (H4): the horizon limits SYNAPTIC LATENCY, not SEQUENCE LENGTH.
"""

from __future__ import annotations

import random

import pytest

from phoenix.cell import Cell
from phoenix.network_graph import Network
from phoenix.synapse import Synapse

A, B, NOISE = 1, 2, 3
SUICIDE_FLOOR = 12.8124
N_TRIALS = 30
SPACING = 300.0


def _syn(pre: int, post: int, weight: float, delay: float, **kwargs: float) -> Synapse:
    return Synapse(
        pre_id=pre, post_id=post, weight=weight, distance=delay,
        propagation_speed=1.0, decay_constant=1000.0, **kwargs,
        tau_decay=1e18,  # [CD] isolate from decay; see test_network_graph._syn
    )


def _drive_pair(synapse: Synapse, latency: float, n_trials: int = N_TRIALS) -> None:
    """Fire pre, then post `latency` ms later, repeatedly, resolving timeouts."""
    for k in range(n_trials):
        base = 100.0 + k * SPACING
        synapse.on_pre_spike(base, None)
        # Advance the clock tick by tick, exactly as Network.step() does, so a
        # pre-spike whose window expires unconfirmed is genuinely scored a miss.
        for t in range(int(base) + 1, int(base + latency) + 1):
            synapse.resolve_timeouts(float(t))
        synapse.on_post_spike(base + latency, base)
        synapse.resolve_timeouts(base + latency + 50.0)


# ---------------------------------------------------------------------------
# H1. The window is DERIVED from tau_stdp, not hardcoded
# ---------------------------------------------------------------------------
def test_verify_window_derives_from_tau_stdp() -> None:
    default = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert default.verify_window_factor == 0.5
    assert default.verify_window == default.verify_window_factor * default.tau_stdp
    assert default.verify_window == 10.0  # 0.5 * 20.0

    # It TRACKS tau_stdp — the whole point. A slower-learning synapse gets a
    # proportionally wider causal horizon.
    slow = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, tau_stdp=40.0)
    assert slow.verify_window == slow.verify_window_factor * slow.tau_stdp
    assert slow.verify_window == 20.0

    # And it stays explicitly overridable (tests and experiments rely on this).
    pinned = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, verify_window=6.0)
    assert pinned.verify_window == 6.0


# ---------------------------------------------------------------------------
# H2. THE BUG THIS STEP FIXES: 8 ms causation is now recognised
# ---------------------------------------------------------------------------
def test_causation_within_horizon_is_recognised() -> None:
    """An 8 ms latency is ordinary biology, and STDP was already learning from it.

    Under the old hardcoded 6 ms window this partner scored causal_success = 0.0,
    had ALL its potentiation gated to zero, and was depressed as if it were noise.
    Inside the derived 10 ms horizon it is correctly recognised as causal.
    """
    synapse = _syn(A, B, weight=5.0, delay=1.0)
    assert 8.0 < synapse.verify_window  # inside the horizon

    _drive_pair(synapse, latency=8.0)

    # Every pre-spike was confirmed: this partner really does cause its post.
    assert synapse.hits == N_TRIALS
    assert synapse.misses == 0
    assert synapse.causal_success == pytest.approx(0.857, abs=0.01)
    assert synapse.causal_success > 0.5

    # So it POTENTIATES, rather than being depressed as noise.
    assert synapse.weight > 5.0


# ---------------------------------------------------------------------------
# H3. Beyond the horizon: rejected (INTENDED)
# ---------------------------------------------------------------------------
def test_causation_beyond_horizon_is_rejected() -> None:
    """A single synapse jumping a 25 ms gap is judged non-causal. This is INTENDED.

    It is the one thing the horizon forbids — and such a synapse was ALREADY
    physically useless: with decay_constant = 10, a 25 ms delay (= 25 distance
    units) attenuates the weight to w * exp(-2.5) ~= 8% of its value. It would
    take dozens of such synapses to fire a single cell. The causal horizon only
    refuses to learn what cable attenuation had already made negligible; the two
    are consistent, not in conflict.
    """
    synapse = _syn(A, B, weight=5.0, delay=1.0)
    assert 25.0 > synapse.verify_window  # beyond the horizon

    _drive_pair(synapse, latency=25.0)

    # The post cell did eventually fire — but far too late to be credited.
    assert synapse.hits == 0
    assert synapse.misses == N_TRIALS
    assert synapse.causal_success == 0.0

    # Potentiation is gated to exactly zero (causal_success = 0), while
    # depression is gated by (1 - causal_success) = 1.0 and so lands in FULL. The
    # synapse therefore does not learn — it is actively, if gently, pruned
    # (measured 4.99999969). A beyond-horizon synapse is treated exactly like
    # noise, which is the intent.
    assert synapse.weight < 5.0                              # never potentiated
    assert synapse.weight == pytest.approx(5.0, abs=1e-5)    # only a whisker


# ---------------------------------------------------------------------------
# H4. THE IMPORTANT ONE: the horizon limits LATENCY, not SEQUENCE LENGTH
# ---------------------------------------------------------------------------
def test_long_sequence_learns_as_a_chain() -> None:
    """A long temporal pattern is learned as a CHAIN of short causal links.

    This is the difference between a harmless constraint and a crippling one. The
    chain 1->2->3->4 spans 9 ms end to end — and a longer chain would exceed the
    10 ms horizon entirely — yet EVERY synapse learns, because no synapse ever
    sees more than its own 3 ms hop. A 500 ms pattern can be learned by 150
    synapses each seeing 3 ms.

    The horizon constrains how far ONE synapse may reach, not how long a sequence
    the NETWORK may represent.
    """
    hop = 3.0
    chain = [(1, 2), (2, 3), (3, 4)]

    net = Network(dt=1.0)
    for neuron_id in (1, 2, 3, 4):
        net.add_cell(Cell(neuron_id=neuron_id))
    for pre, post in chain:
        net.add_synapse(_syn(pre, post, weight=5.0, delay=1.0))

    # Drive the sequence: each cell fires `hop` ms after the previous one.
    for k in range(N_TRIALS):
        base = 100 + k * int(SPACING)
        while net.current_time < base + 20:
            for index, neuron_id in enumerate((1, 2, 3, 4)):
                if net.current_time == base - 1 + hop * index:
                    net.inject(neuron_id, 100.0)
            net.step()

    horizon = net.outgoing[1][0].verify_window
    total_span = hop * len(chain)  # 9 ms end-to-end: nearly the whole horizon,
    assert total_span == 9.0       # and one more hop would exceed it entirely.

    # What actually matters is that each individual HOP is inside the horizon —
    # the end-to-end span is irrelevant to any single synapse.
    assert hop < horizon

    # EVERY link in the chain learned.
    for pre, post in chain:
        synapse = next(s for s in net.outgoing[pre] if s.post_id == post)
        assert synapse.causal_success is not None
        assert synapse.causal_success > 0.5
        assert synapse.causal_success == pytest.approx(0.857, abs=0.01)
        assert synapse.weight > 5.0  # potentiated


# ---------------------------------------------------------------------------
# H5. The three CST criteria still hold at the wider 10 ms window
# ---------------------------------------------------------------------------
def test_noise_discrimination_survives_the_wider_window() -> None:
    """Loop alive, weight interior, noise pruned — with verify_window now 10 ms.

    Noise level 5%: this is the lowest level in the sweep at which the widened
    window's discrimination is verified. See the module-level report and
    test_causal_success for the measured tradeoff — at 2% noise the 10 ms window
    does NOT separate the weights (gap -0.002, versus +0.516 at 6 ms).
    """
    net = Network(dt=1.0)
    for neuron_id in (A, B, NOISE):
        net.add_cell(Cell(neuron_id=neuron_id))
    for pre, post in ((A, B), (B, A)):
        net.add_synapse(_syn(pre, post, weight=15.0, delay=1.0))
        net.add_synapse(_syn(pre, post, weight=15.0, delay=2.0))
    net.add_synapse(_syn(NOISE, B, weight=15.0, delay=1.0))

    rng = random.Random(42)
    net.inject(A, 100.0)  # the ONE external ignition
    spikes = []
    for _ in range(30_000):
        if rng.random() < 0.05:
            net.inject(NOISE, 100.0)
        spikes.extend(net.step())

    loop = net.outgoing[A][0]
    noise = net.outgoing[NOISE][0]

    # Criterion 1: still firing at the end of a 30 s run.
    assert any(s.timestamp > 29_000 for s in spikes)

    # Criterion 2: the noise synapse is pruned; the loop is not.
    assert noise.weight < loop.weight
    assert noise.causal_success < 0.5 < loop.causal_success

    # Criterion 3: the loop's weight is INTERIOR — not dead, not welded to w_max.
    assert SUICIDE_FLOOR < loop.weight < loop.w_max - 0.5
    assert abs(loop.weight - loop.w_max) > 1e-3

    # And the horizon in force really is the new derived one.
    assert loop.verify_window == 10.0
