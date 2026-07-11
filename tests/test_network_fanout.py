"""Fan-out (1->N) validation of the general N-cell ``Network``.

No production code accompanies this file: ``Network.step()`` already fans out
(it iterates EVERY synapse in ``outgoing[spike.neuron_id]`` and enqueues an
independent delivery per edge). These tests exercise that behavior, which the
existing suite wired but never validated behaviorally.

Scope, as with the fan-in step, is structure/plumbing only — homeostasis,
per-cell monitors and spontaneous activity are not wired into ``step()``.

Timing convention (same as the fan-in suite): a spike at ``t`` over a synapse
of delay ``d`` arrives at ``t + d``, and delivery is coarsened to the tick whose
``[current_time, current_time + dt)`` contains it — so the earliest possible
arrival is one tick after firing. Expected arrival ticks and attenuated weights
below are DERIVED (delay + latency; ``weight * exp(-distance/decay_constant)``),
not hardcoded.
"""

from __future__ import annotations

import math

import pytest

from phoenix.cell import Cell
from phoenix.network_graph import Network
from phoenix.synapse import Synapse

A, B, C, D = 1, 2, 3, 4  # neuron ids
VREST = -75.0


def _network(*neuron_ids: int, dt: float = 1.0) -> Network:
    net = Network(dt=dt)
    for neuron_id in sorted(neuron_ids):
        net.add_cell(Cell(neuron_id=neuron_id))
    return net


def _syn(
    pre: int, post: int, weight: float, delay: float, decay: float = 1000.0
) -> Synapse:
    """Synapse with delay == `delay`.

    ``decay`` defaults high so cable attenuation is negligible and `weight` is
    (near enough) the mV bump actually delivered — keeping the arithmetic in
    most tests readable. test_fan_out_attenuation_scales_with_distance
    deliberately overrides it with a REAL decay constant.
    """
    return Synapse(
        pre_id=pre, post_id=post, weight=weight,
        distance=delay, propagation_speed=1.0, decay_constant=decay,
    )


def _fire_once(net: Network, neuron_id: int, current: float = 100.0) -> None:
    """Inject a suprathreshold current so `neuron_id` fires on the next step."""
    net.inject(neuron_id, current)


# ---------------------------------------------------------------------------
# 1. Wiring, from the SOURCE side (mirror of the fan-in wiring test)
# ---------------------------------------------------------------------------
def test_fan_out_wiring() -> None:
    net = _network(A, B, C, D)
    syn_ab = _syn(A, B, weight=15.0, delay=1.0)
    syn_ac = _syn(A, C, weight=15.0, delay=1.0)
    syn_ad = _syn(A, D, weight=15.0, delay=1.0)
    for synapse in (syn_ab, syn_ac, syn_ad):
        net.add_synapse(synapse)

    # One source, three outgoing edges, in insertion order.
    assert net.outgoing[A] == [syn_ab, syn_ac, syn_ad]

    # Each target collects exactly its own edge.
    assert net.incoming[B] == [syn_ab]
    assert net.incoming[C] == [syn_ac]
    assert net.incoming[D] == [syn_ad]

    # The targets are leaves.
    assert net.outgoing[B] == []
    assert net.outgoing[C] == []
    assert net.outgoing[D] == []


# ---------------------------------------------------------------------------
# 2. THE core fan-out property: one spike -> N independent deliveries
# ---------------------------------------------------------------------------
def test_one_spike_delivers_to_all_targets() -> None:
    net = _network(A, B, C, D)
    for target in (B, C, D):
        net.add_synapse(_syn(A, target, weight=15.0, delay=1.0))

    _fire_once(net, A)  # A fires exactly once, at t=1

    # Track PEAK Vm per target: the bump leaks away after it lands, so the
    # end-of-run Vm would understate it.
    spikes: list = []
    peak = {target: -1e9 for target in (B, C, D)}
    for _ in range(10):
        spikes.extend(net.step())
        for target in (B, C, D):
            peak[target] = max(peak[target], net.cells[target].Vm)

    # A fired once and nothing else did (15 mV is subthreshold for the targets).
    assert [s.neuron_id for s in spikes] == [A]

    # That ONE spike reached every target independently.
    expected_bump = net.outgoing[A][0].effective_weight()  # ~14.985 mV
    for target in (B, C, D):
        trace_context = net.cells[target].trace_context
        assert trace_context is not None
        source_id, arrived_at = trace_context
        assert source_id == A
        # receive_input runs before the delivering tick's integrate, so the
        # recorded time is (arrival - dt) = (1 + 1) - 1 = 1.
        assert arrived_at == 1.0

        # Exactly one bump landed: Vm peaked one weight above rest.
        assert peak[target] == pytest.approx(VREST + expected_bump, abs=0.05)


# ---------------------------------------------------------------------------
# 3. Staggered arrivals: one spike, three different delays
# ---------------------------------------------------------------------------
def test_fan_out_different_delays_arrive_at_different_times() -> None:
    net = _network(A, B, C, D)
    delays = {B: 1.0, C: 5.0, D: 10.0}
    for target, delay in sorted(delays.items()):
        net.add_synapse(_syn(A, target, weight=15.0, delay=delay))

    _fire_once(net, A)  # A fires at t=1

    first_arrival: dict[int, float] = {}
    for _ in range(20):
        net.step()
        for target in sorted(delays):
            if target not in first_arrival:
                if net.cells[target].trace_context is not None:
                    first_arrival[target] = net.current_time

    fire_time = 1.0
    for target, delay in sorted(delays.items()):
        # Arrival = fire_time + delay, realized on the tick that contains it.
        expected = fire_time + delay
        assert first_arrival[target] == expected

    # Concretely: B at t=2, C at t=6, D at t=11 — one spike, staggered fan-out.
    assert first_arrival == {B: 2.0, C: 6.0, D: 11.0}


# ---------------------------------------------------------------------------
# 4. Cable attenuation through the real delivery path (decay kept LIVE here)
# ---------------------------------------------------------------------------
def test_fan_out_attenuation_scales_with_distance() -> None:
    decay = 10.0
    weight = 15.0
    near, far = 1.0, 5.0

    net = _network(A, B, C)
    syn_ab = _syn(A, B, weight=weight, delay=near, decay=decay)  # B is NEARER
    syn_ac = _syn(A, C, weight=weight, delay=far, decay=decay)   # C is FARTHER
    net.add_synapse(syn_ab)
    net.add_synapse(syn_ac)

    # Derived, not hardcoded: weight * exp(-distance / decay_constant).
    expected_near = weight * math.exp(-near / decay)  # ~13.573 mV
    expected_far = weight * math.exp(-far / decay)    # ~9.098 mV
    assert syn_ab.effective_weight() == pytest.approx(expected_near)
    assert syn_ac.effective_weight() == pytest.approx(expected_far)
    assert expected_near > expected_far

    _fire_once(net, A)

    # Peak Vm per target == its single delivered bump above rest (no other input,
    # and each stays subthreshold so nothing resets).
    peak = {B: -1e9, C: -1e9}
    for _ in range(20):
        net.step()
        for target in (B, C):
            peak[target] = max(peak[target], net.cells[target].Vm)

    rise_near = peak[B] - VREST
    rise_far = peak[C] - VREST

    # The nearer target receives a strictly larger bump — attenuation survives
    # the whole propagate -> enqueue -> receive_input path, not just the formula.
    assert rise_near > rise_far
    assert rise_near == pytest.approx(expected_near, abs=0.05)
    assert rise_far == pytest.approx(expected_far, abs=0.05)


# ---------------------------------------------------------------------------
# 5. One presynaptic cell, several outgoing synapses evolving INDEPENDENTLY
# ---------------------------------------------------------------------------
def test_fan_out_stdp_independent_per_synapse() -> None:
    net = _network(A, B, C)
    net.add_synapse(_syn(A, B, weight=5.0, delay=1.0))
    net.add_synapse(_syn(A, C, weight=5.0, delay=1.0))

    # Per trial:  C fires at base,  A fires at base+5,  B fires at base+7.
    #   A fires AFTER  C  -> A->C is anti-causal -> DEPRESS
    #   A fires BEFORE B  -> A->B is causal      -> POTENTIATE
    # So the SAME presynaptic spikes drive A's two outgoing synapses in OPPOSITE
    # directions, decided purely by each one's own postsynaptic partner timing.
    #
    # Trials are spaced 300 ms (>> tau_stdp=20 ms), so cross-trial trace bleed is
    # ~exp(-300/20) ~= 3e-7. Incidental terms (e.g. on_pre_spike for A->B firing
    # against a stale post_trace ~= 0) are small but NOT exactly zero, which is
    # why only strict inequalities are asserted below.
    n_trials, spacing = 30, 300
    for k in range(n_trials):
        base = 100 + k * spacing
        inject_c = base - 1        # -> C fires at base
        inject_a = base + 4        # -> A fires at base + 5
        inject_b = base + 6        # -> B fires at base + 7
        while net.current_time < base + 12:
            if net.current_time == inject_c:
                _fire_once(net, C)
            if net.current_time == inject_a:
                _fire_once(net, A)
            if net.current_time == inject_b:
                _fire_once(net, B)
            net.step()

    # Verified: weight(A,B) = 5.03529, weight(A,C) = 4.76636.
    assert net.weight(A, B) > 5.0   # causal partner  -> potentiated
    assert net.weight(A, C) < 5.0   # anti-causal one -> depressed
    assert net.weight(A, B) > net.weight(A, C)


# ---------------------------------------------------------------------------
# 6. Deliveries to different targets are mutually independent
# ---------------------------------------------------------------------------
def test_fan_out_targets_are_independent_on_delivery() -> None:
    net = _network(A, B, C)
    net.add_synapse(_syn(A, B, weight=15.0, delay=1.0))
    net.add_synapse(_syn(A, C, weight=15.0, delay=1.0))

    # A and B both fire at t=1. A's bumps arrive at t=2 — but B set
    # refractory_until = 1 + refractory_period = 3 when it fired, and delivery
    # calls receive_input while B.t is still 1 (< 3), so B's bump is HARD
    # REJECTED (Option A). C, at rest, must be unaffected.
    _fire_once(net, A)
    _fire_once(net, B)

    # Peak-track again: the delivered bump decays after it lands.
    spikes: list = []
    peak_b, peak_c = -1e9, -1e9
    for _ in range(6):
        spikes.extend(net.step())
        peak_b = max(peak_b, net.cells[B].Vm)
        peak_c = max(peak_c, net.cells[C].Vm)

    assert sorted(s.neuron_id for s in spikes) == [A, B]  # both fired once, at t=1

    # C received A's bump normally.
    c_context = net.cells[C].trace_context
    assert c_context is not None
    assert c_context[0] == A
    assert peak_c == pytest.approx(
        VREST + net.outgoing[A][1].effective_weight(), abs=0.05
    )

    # B did NOT: its trace_context still shows only the (source-less) external
    # injection, never A, and its Vm never rose above Vreset — the bump was
    # dropped outright rather than merely delayed.
    b_context = net.cells[B].trace_context
    assert b_context is not None
    assert b_context[0] is None  # never overwritten by A's rejected delivery
    assert peak_b == pytest.approx(net.cells[B].Vreset, abs=1e-9)

    # One target being blocked did not disturb the other: fan-out deliveries are
    # independent per edge.
