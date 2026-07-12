"""Recurrence (loops) on the general ``Network``.

Every prior test is FEED-FORWARD. These run the first cycles: A<->B, self-loops,
self-sustained activity, and the same-tick double-update property that was
flagged during the fan-in review but never verified.

No production code accompanies this file — the ``Network``'s adjacency maps
impose no acyclicity, so cycles already work structurally.

>>> THE STRUCTURAL CONSTRAINT EVERYTHING HERE IS DESIGNED AROUND <<<
A single synapse CANNOT fire a resting cell:

    gap to threshold = Vthresh - Vrest = -50 - (-75) = 25.0 mV
    synapse ceiling  = w_max                          = 20.0 mV
    => 20.0 < 25.0

So a naive one-synapse-per-direction loop can never reverberate, and raising the
weight does not help (``clamp(w, w_min, w_max)`` cuts it back to 20). This is not
a defect. Recurrence must therefore be built ON TOP OF fan-in: every cell in a
cycle needs CONVERGENT input — >= 2 synapses arriving close enough together to
sum past threshold. No frozen constant (w_max / Vthresh / Vrest /
refractory_period / learning_rate) is retuned anywhere in this file; the loops
below are built purely from convergence.
"""

from __future__ import annotations

from collections import Counter

import pytest

from phoenix.cell import Cell
from phoenix.network_graph import Network
from phoenix.spike import Spike
from phoenix.synapse import Synapse

A, B = 1, 2
VREST, VTHRESH = -75.0, -50.0
W_MAX = 20.0


def _network(*neuron_ids: int, dt: float = 1.0) -> Network:
    net = Network(dt=dt)
    for neuron_id in sorted(neuron_ids):
        net.add_cell(Cell(neuron_id=neuron_id))
    return net


def _syn(
    pre: int, post: int, weight: float, delay: float, decay: float = 1000.0
) -> Synapse:
    """Synapse with delay == `delay`; `decay` high so `weight` is the mV bump."""
    return Synapse(
        pre_id=pre, post_id=post, weight=weight, distance=delay,
        propagation_speed=1.0, decay_constant=decay,
        tau_decay=1e18,  # [CD] isolate from decay; see test_network_graph._syn
    )


def _train(spikes: list[Spike]) -> list[tuple[int, float]]:
    return [(s.neuron_id, s.timestamp) for s in spikes]


# ---------------------------------------------------------------------------
# R1. A cycle is structurally accepted — no acyclicity assumption anywhere
# ---------------------------------------------------------------------------
def test_cycle_wiring_is_accepted() -> None:
    net = _network(A, B)
    syn_ab = _syn(A, B, weight=15.0, delay=1.0)
    syn_ba = _syn(B, A, weight=15.0, delay=1.0)

    # Neither call raises: add_synapse makes no acyclicity check.
    net.add_synapse(syn_ab)
    net.add_synapse(syn_ba)

    # The adjacency maps represent the cycle from both sides.
    assert net.outgoing[A] == [syn_ab]
    assert net.incoming[A] == [syn_ba]
    assert net.outgoing[B] == [syn_ba]
    assert net.incoming[B] == [syn_ab]


# ---------------------------------------------------------------------------
# R2. A single-synapse loop CANNOT reverberate — pinning the structural fact
# ---------------------------------------------------------------------------
def test_single_synapse_loop_cannot_reverberate() -> None:
    """This is NOT a defect. It is a structural consequence of frozen constants.

    A lone synapse is capped at w_max = 20 mV, but a resting cell needs
    Vthresh - Vrest = 25 mV to fire. 20 < 25, so one synapse can never ignite the
    next cell in a loop, and raising its weight cannot help — the clamp cuts it
    back to 20. Loops therefore REQUIRE convergent input (see R3).

    This test pins that fact so it can never be silently "fixed" by retuning
    w_max / Vthresh / Vrest.
    """
    net = _network(A, B)
    # Both synapses at the ceiling — the strongest a single edge can ever be.
    net.add_synapse(_syn(A, B, weight=W_MAX, delay=1.0))
    net.add_synapse(_syn(B, A, weight=W_MAX, delay=1.0))

    net.inject(A, 100.0)  # ignite the loop ONCE
    spikes = net.run(n_steps=50)

    # The activity dies immediately. Measured: exactly ONE spike (A at t=1); B
    # never fires at all, because a 20 mV bump only lifts it to ~-55 mV, still
    # 5 mV short of the -50 mV threshold.
    assert len(spikes) <= 2
    assert _train(spikes) == [(A, 1.0)]

    # Nothing at all after the ignition transient — no reverberation.
    assert all(s.timestamp <= 5.0 for s in spikes)

    # And the reason, asserted directly: one edge cannot bridge the gap.
    assert W_MAX < (VTHRESH - VREST)


# ---------------------------------------------------------------------------
# R3. THE core recurrence test: convergent input sustains a loop
# ---------------------------------------------------------------------------
def test_loop_reverberates_with_convergent_input() -> None:
    """One injection -> self-sustained activity, built purely from CONVERGENCE.

    Topology: a 2-cell cycle with every edge DOUBLED — two synapses per
    direction (weight 15 mV each, delays 1 ms and 2 ms). Neither bump alone can
    fire the target (15 < 25), but the two arrive 1 ms apart and SUM past
    threshold, exactly as validated in the fan-in step. That convergence, not any
    retuned constant, is what carries the loop.

    Measured: ONE injection at t=0 yields a perfectly regular alternating train
    (A@1, B@3, A@5, B@7, ...) — a 4 ms period, 500 Hz aggregate — versus a
    feed-forward baseline of just 2 spikes.

    HISTORY: this test originally pinned the loop's SUICIDE (108 spikes, dead at
    t=215). Causal Success Tracking [CST] fixed that; it now runs the full 400 ms
    (200 spikes) and keeps going. See the assertions below and
    tests/test_causal_success.py.
    """
    net = _network(A, B)
    for pre, post in ((A, B), (B, A)):
        net.add_synapse(_syn(pre, post, weight=15.0, delay=1.0))
        net.add_synapse(_syn(pre, post, weight=15.0, delay=2.0))

    # The ONLY external drive, at t0. Nothing is injected ever again.
    net.inject(A, 100.0)
    spikes = net.run(n_steps=400)

    train = _train(spikes)
    counts = Counter(neuron_id for neuron_id, _ in train)

    # Feed-forward baseline: the same convergent pair WITHOUT the return edges,
    # i.e. the same initial pulse propagating exactly once.
    ff = _network(A, B)
    ff.add_synapse(_syn(A, B, weight=15.0, delay=1.0))
    ff.add_synapse(_syn(A, B, weight=15.0, delay=2.0))
    ff.inject(A, 100.0)
    baseline = ff.run(n_steps=400)
    assert len(baseline) == 2  # A fires, B fires, done.

    # 1) Activity is genuinely REGENERATED by the loop, not just one pulse.
    assert len(spikes) > 10 * len(baseline)

    # 2) Reverberation's signature: cells fire many times, not once.
    assert counts[A] > 1 and counts[B] > 1

    # 3) It continues LONG after the external drive stopped. The feed-forward
    #    version is silent by t=3; this is still firing at t>100.
    assert max(t for _, t in train) > 100.0

    # 4) The loop alternates cleanly at a 4 ms period (2 ms per hop).
    assert train[:6] == [(A, 1.0), (B, 3.0), (A, 5.0), (B, 7.0), (A, 9.0), (B, 11.0)]

    # --- [CST] THE LOOP NO LONGER COMMITS SUICIDE ---
    # This test originally PINNED the bug: raw STDP is net-depressing under
    # recurrence (in a cycle every synapse is anti-causal to ITSELF on the next
    # lap), so weights slid 15.0 -> 12.835, crossed the ~12.82 sustain floor, and
    # the loop fell silent at t=215 after 108 spikes.
    #
    # Causal Success Tracking fixes exactly that: the loop's synapses reliably
    # cause their post cells (causal_success ~1.0), so their depression is gated
    # by (1 - causal_success) ~ 0 and they are PROTECTED from the loop's own
    # anti-causal self-depression. The loop now fires for the whole run.
    assert len(spikes) == 200                       # was 108, then dead
    assert counts[A] == 100 and counts[B] == 100
    assert max(t for _, t in train) > 390.0         # still firing at the horizon
    assert any(t > 250.0 for _, t in train)         # was: silent after 215

    # Weights stay ABOVE the suicide floor (~12.8124) rather than sliding under it.
    final_ab = [s.weight for s in net.outgoing[A]]
    final_ba = [s.weight for s in net.outgoing[B]]
    assert all(w > 12.8124 for w in final_ab + final_ba)

    # And the loop knows it is causal — which is precisely what protects it.
    assert net.outgoing[A][0].causal_success > 0.9


# ---------------------------------------------------------------------------
# R4. The flagged property: same-tick double update on a recurrent synapse
# ---------------------------------------------------------------------------
def test_same_tick_double_update_on_recurrent_synapse() -> None:
    """A synapse whose endpoints fire in the SAME tick takes BOTH trace updates.

    In a cycle, X->Y can have both endpoints fire on one tick. step() then calls
    on_pre_spike (X is pre) AND on_post_spike (Y is post) on that single synapse
    within one step.

    What this means: the synapse records NO causal observation — correct, since
    simultaneity is not causation (delta_t = 0, which record_observation rejects
    as non-causal) — but it DOES take both trace increments, and the potentiation
    half sees the pre_trace that the depression half just incremented. This is
    inherited from TwoCellNetwork's ordering and is a MODELING CHOICE, not an
    accident.
    """
    net = _network(A, B)
    net.add_synapse(_syn(A, B, weight=5.0, delay=1.0))

    # Force BOTH endpoints of A->B to fire on the same tick.
    net.inject(A, 100.0)
    net.inject(B, 100.0)
    spikes = net.step()

    assert _train(spikes) == [(A, 1.0), (B, 1.0)]  # same timestamp

    synapse = net.outgoing[A][0]

    # Both traces incremented — the synapse saw itself as pre AND as post.
    assert synapse.pre_trace == pytest.approx(1.0)
    assert synapse.post_trace == pytest.approx(1.0)

    # No causal observation: simultaneity is not causation.
    assert synapse.observed_delays == []

    # The net weight change, pinned exactly. on_pre_spike runs first (spikes are
    # ordered by neuron_id) and depresses by post_trace, which is still 0 -> dw=0.
    # It then sets pre_trace=1. on_post_spike then potentiates by that pre_trace:
    #   dw = learning_rate * modulation * A_plus * pre_trace
    #      = 0.01 * 1.0 (no expectation yet -> neutral) * 1.0 * 1.0 = +0.01
    assert synapse.weight == pytest.approx(5.01, abs=1e-9)


# ---------------------------------------------------------------------------
# R5. Raw STDP structurally PENALIZES a loop's return path
# ---------------------------------------------------------------------------
def test_recurrent_synapse_is_depressed_by_anticausal_return() -> None:
    """The return edge of a loop is depressed while the forward edge is potentiated.

    From the ignition spike's point of view A fires, then B. For the FORWARD edge
    A->B that ordering is causal (pre before post) -> potentiation. For the RETURN
    edge B->A the very same ordering is ANTI-causal (its post, A, fired before its
    pre, B) -> depression.

    So raw STDP structurally penalizes the return path of any loop. Whether that
    is desirable is a DESIGN QUESTION; this test only measures and pins it.
    Measured after 30 well-separated trials: A->B = 5.035289, B->A = 4.728549.
    """
    net = _network(A, B)
    net.add_synapse(_syn(A, B, weight=5.0, delay=1.0))
    net.add_synapse(_syn(B, A, weight=5.0, delay=1.0))

    # Drive A then B externally, 2 ms apart, so the loop ordering is imposed
    # cleanly without relying on reverberation. Trials 300 ms apart (>> tau_stdp
    # = 20 ms) so no traces bleed across trials.
    for k in range(30):
        base = 100 + k * 300
        while net.current_time < base + 10:
            if net.current_time == base - 1:
                net.inject(A, 100.0)      # A fires at `base`
            if net.current_time == base + 1:
                net.inject(B, 100.0)      # B fires at `base` + 2
            net.step()

    forward = net.weight(A, B)
    return_path = net.weight(B, A)

    # Strict inequalities only — the asymmetry is the finding, not its magnitude.
    assert forward > 5.0            # causal      -> potentiated
    assert return_path < 5.0        # anti-causal -> depressed
    assert forward > return_path


# ---------------------------------------------------------------------------
# R6. Safety: a cycle cannot re-enter within a single step()
# ---------------------------------------------------------------------------
def test_no_infinite_loop_within_single_step() -> None:
    """Delivery is always deferred by >= 1 tick, so a loop cannot recurse in a step.

    A self-loop A->A is the tightest possible cycle. Its spike's effect on its own
    cell must appear no earlier than t + delay — never inside the firing tick.
    The _pending queue guarantees this: a spike enqueued during step() has
    arrival >= this tick's end, so it can only be delivered on a LATER tick. That
    is what prevents zero-delay infinite recursion.

    (delay=5 ms is used, not 1 ms: at delay=1 the returning bump arrives at t=2,
    inside A's own refractory window (refractory_until = 3), so Option A hard-
    rejects it and there would be nothing to observe. 5 ms clears refractory.)
    """
    delay = 5.0
    net = _network(A)
    net.add_synapse(_syn(A, A, weight=15.0, delay=delay))

    net.inject(A, 100.0)

    # The firing tick itself: step() terminates, and A's own spike has NOT fed
    # back into A within this same tick.
    spikes = net.step()
    assert _train(spikes) == [(A, 1.0)]
    fire_time = 1.0
    assert net.cells[A].Vm == pytest.approx(net.cells[A].Vreset)  # reset, no self-input
    source, _ = net.cells[A].trace_context                        # only the injection
    assert source is None                                          # NOT A: no re-entry

    # Advance and find the first tick at which A's own spike reaches A.
    self_input_at: float | None = None
    for _ in range(10):
        net.step()
        context = net.cells[A].trace_context
        if self_input_at is None and context is not None and context[0] == A:
            self_input_at = net.current_time

    # The effect appears exactly at fire_time + delay, and never earlier.
    assert self_input_at == fire_time + delay
    assert self_input_at > fire_time  # strictly later: no same-tick recursion
