"""Fan-in (2->1) validation of the general N-cell ``Network``.

Scope: this validates the STRUCTURE/plumbing of fan-in — adjacency wiring,
delayed delivery, temporal summation, arrival-vs-firing order, and fan-in STDP.
Homeostasis / per-cell monitors / spontaneous activity are deliberately NOT
wired into ``Network.step()`` yet (same as ``TwoCellNetwork``), so they are not
exercised here.

All timings below are derived from ACTUAL arrival times (spike_time +
synapse.delay) and were calibrated against the real ``Cell``, not guessed.
Delivery has a minimum one-tick latency and is coarsened to tick boundaries,
and the delivering tick calls ``receive_input`` BEFORE ``integrate``.
"""

from __future__ import annotations

import pytest

from phoenix.cell import Cell
from phoenix.network_graph import Network
from phoenix.synapse import Synapse

A, B, C = 1, 2, 3  # neuron ids


def _fan_in_network(dt: float = 1.0) -> Network:
    """Three cells (A, B, C) registered, no synapses yet."""
    net = Network(dt=dt)
    net.add_cell(Cell(neuron_id=A))
    net.add_cell(Cell(neuron_id=B))
    net.add_cell(Cell(neuron_id=C))
    return net


def _syn(pre: int, post: int, weight: float, delay: float) -> Synapse:
    """Synapse whose effective_weight ~= weight and whose delay == `delay`.

    decay_constant is set far above the distance so cable attenuation is
    negligible and `weight` is the mV bump actually delivered — keeping the
    summation arithmetic below readable.
    """
    return Synapse(
        pre_id=pre, post_id=post, weight=weight,
        distance=delay, propagation_speed=1.0, decay_constant=1000.0,
    )


# ---------------------------------------------------------------------------
# 1. Wiring
# ---------------------------------------------------------------------------
def test_add_cell_and_synapse_wiring() -> None:
    net = _fan_in_network()
    syn_ac = _syn(A, C, weight=15.0, delay=1.0)
    syn_bc = _syn(B, C, weight=15.0, delay=2.0)
    net.add_synapse(syn_ac)
    net.add_synapse(syn_bc)

    # Fan-out side: each source cell has exactly its own outgoing edge.
    assert net.outgoing[A] == [syn_ac]
    assert net.outgoing[B] == [syn_bc]
    assert net.outgoing[C] == []

    # Fan-in side: C collects BOTH incoming edges — this is what TwoCellNetwork
    # structurally cannot represent.
    assert net.incoming[C] == [syn_ac, syn_bc]
    assert net.incoming[A] == []
    assert net.incoming[B] == []

    # Dangling connections fail loud, on either endpoint.
    with pytest.raises(ValueError):
        net.add_synapse(_syn(A, 99, weight=1.0, delay=1.0))
    with pytest.raises(ValueError):
        net.add_synapse(_syn(99, C, weight=1.0, delay=1.0))


# ---------------------------------------------------------------------------
# 2. Identity must be unique
# ---------------------------------------------------------------------------
def test_duplicate_neuron_id_rejected() -> None:
    net = Network(dt=1.0)
    net.add_cell(Cell(neuron_id=A))
    with pytest.raises(ValueError):
        net.add_cell(Cell(neuron_id=A))


# ---------------------------------------------------------------------------
# 3. Temporal summation: two subthreshold arrivals together cross threshold
# ---------------------------------------------------------------------------
def test_temporal_integration_two_inputs_sum_to_spike() -> None:
    net = _fan_in_network()
    # C needs +25 mV (Vrest -75 -> Vthresh -50). Each bump is 15 mV: subthreshold
    # alone (see test 5), sufficient together IF the first has not yet leaked
    # away. With tau=20ms, a 15 mV bump still contributes >=10 mV for
    # dt <= 20*ln(1.5) ~= 8.1 ms, so the two ARRIVALS must land within ~8 ms.
    net.add_synapse(_syn(A, C, weight=15.0, delay=1.0))
    net.add_synapse(_syn(B, C, weight=15.0, delay=2.0))

    # Both fire at t=1 -> arrivals at t=2 (A) and t=3 (B): a 1 ms arrival gap.
    net.inject(A, 100.0)
    net.inject(B, 100.0)

    c_spikes = [
        s.timestamp
        for _ in range(10)
        for s in net.step()
        if s.neuron_id == C
    ]

    # The summed arrivals cross threshold: C fires on the tick the second
    # (B's) bump lands, once the first has only leaked one tick.
    assert c_spikes == [3.0]


# ---------------------------------------------------------------------------
# 4. Arrival order != firing order (the core causal property)
# ---------------------------------------------------------------------------
def test_arrival_order_differs_from_firing_order_under_delay() -> None:
    net = _fan_in_network()
    # A is FAR (delay 20), B is NEAR (delay 1). Both bumps stay subthreshold so
    # C never fires/resets and we can watch each bump land.
    net.add_synapse(_syn(A, C, weight=15.0, delay=20.0))
    net.add_synapse(_syn(B, C, weight=15.0, delay=1.0))

    fire_a = fire_b = None
    first_arrival: tuple[float, int] | None = None

    for _ in range(30):
        if net.current_time == 0.0:
            net.inject(A, 100.0)   # A fires FIRST, at t=1
        if net.current_time == 4.0:
            net.inject(B, 100.0)   # B fires LATER, at t=5
        for spike in net.step():
            if spike.neuron_id == A:
                fire_a = spike.timestamp
            if spike.neuron_id == B:
                fire_b = spike.timestamp
        # Record which source's input first reached C.
        if first_arrival is None and net.cells[C].trace_context is not None:
            source_id, _ = net.cells[C].trace_context
            first_arrival = (net.current_time, source_id)

    # A fired BEFORE B...
    assert fire_a == 1.0
    assert fire_b == 5.0
    assert fire_a < fire_b

    # ...but because A is far and B is near, B's contribution ARRIVES first.
    arrival_a = fire_a + net.outgoing[A][0].delay   # 1 + 20 = 21
    arrival_b = fire_b + net.outgoing[B][0].delay   # 5 +  1 =  6
    assert arrival_b < arrival_a

    # And the cell actually sees it that way: the first input to reach C came
    # from B, not from the earlier-firing A. "Who fired first" != "who arrived
    # first" — geometry, not firing order, decides.
    assert first_arrival is not None
    arrived_at, arrived_from = first_arrival
    assert arrived_from == B
    assert arrived_at == 6.0


# ---------------------------------------------------------------------------
# 5. One input alone is genuinely insufficient (test 3 really needed both)
# ---------------------------------------------------------------------------
def test_single_input_insufficient_no_spike() -> None:
    net = _fan_in_network()
    net.add_synapse(_syn(A, C, weight=15.0, delay=1.0))
    net.add_synapse(_syn(B, C, weight=15.0, delay=2.0))

    net.inject(A, 100.0)  # only A fires; B stays silent

    peak_vm = -1e9
    c_spiked = False
    for _ in range(15):
        for spike in net.step():
            if spike.neuron_id == C:
                c_spiked = True
        peak_vm = max(peak_vm, net.cells[C].Vm)

    # A lone 15 mV bump peaks ~-60 mV and leaks back toward Vrest: it never
    # reaches Vthresh (-50). So the spike in test 3 was genuinely produced by
    # SUMMATION of two inputs, not by either one alone.
    assert not c_spiked
    assert peak_vm < net.cells[C].Vthresh
    assert peak_vm == pytest.approx(-60.0, abs=0.1)


# ---------------------------------------------------------------------------
# 6. Fan-in STDP: the partner that fires CLOSER to C potentiates more
# ---------------------------------------------------------------------------
def test_fan_in_stdp_closer_partner_potentiates_more() -> None:
    net = _fan_in_network()
    net.add_synapse(_syn(A, C, weight=5.0, delay=1.0))
    net.add_synapse(_syn(B, C, weight=5.0, delay=1.0))

    # Each trial: B fires at T-8, A fires at T-2, C fires at T. So A's spike is
    # CLOSER to C's than B's is (dt 2 ms vs 8 ms) -> at C's firing the A->C
    # pre_trace (exp(-2/tau_stdp)) exceeds the B->C one (exp(-8/tau_stdp)), so
    # on_post_spike potentiates A->C more. A clear gap (2 vs 8), not two close
    # values, keeps the effect unambiguous.
    #
    # Trials are spaced 300 ms apart (>> tau_stdp=20 ms) so traces from one
    # trial have decayed to ~exp(-15) before the next begins — no cross-trial
    # bleed. Note that when A/B fire, on_pre_spike also runs against C's STALE
    # last_spike_time, applying a depression term scaled by post_trace ~= 0
    # after such a gap: negligible, and it does not corrupt the comparison.
    n_trials, spacing = 30, 300
    for k in range(n_trials):
        t_c = 100 + k * spacing        # C fires here
        inject_b = t_c - 9             # -> B fires at t_c - 8
        inject_a = t_c - 3             # -> A fires at t_c - 2
        inject_c = t_c - 1             # -> C fires at t_c
        while net.current_time < t_c + 5:
            if net.current_time == inject_b:
                net.inject(B, 100.0)
            if net.current_time == inject_a:
                net.inject(A, 100.0)
            if net.current_time == inject_c:
                net.inject(C, 100.0)
            net.step()

    # Strict inequality only — asserting a magnitude threshold on the difference
    # would be brittle. THE CORE CLAIM, unchanged: the partner that fires closer
    # to C potentiates more.
    assert net.weight(A, C) > net.weight(B, C)

    # BOTH partners potentiate: this is a timing ASYMMETRY, not one-sided learning.
    #
    # [CST-2] History: under CST's hardcoded 6 ms verify_window, B (firing 8 ms
    # before C) fell OUTSIDE the causal horizon, was scored a full MISS, and was
    # net-DEPRESSED as if it were noise — even though STDP itself was learning
    # from it (tau_stdp = 20 ms). That inconsistency is exactly what CST-2 fixed:
    # the horizon is now derived (0.5 * tau_stdp = 10 ms), so B's 8 ms latency is
    # correctly recognised as causal (causal_success 0.857, same as A's) and it
    # potentiates again. A still potentiates MORE, because its pre_trace at C's
    # firing is larger — which is the actual claim of this test.
    assert net.weight(A, C) > 5.0
    assert net.weight(B, C) > 5.0
