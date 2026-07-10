"""Tests for the Phoenix ``TwoCellNetwork`` minimal simulation loop."""

import math

from phoenix.cell import Cell
from phoenix.network import TwoCellNetwork
from phoenix.synapse import Synapse


def test_no_spikes_with_zero_input() -> None:
    cell_a = Cell(neuron_id=1)
    cell_b = Cell(neuron_id=2)
    synapse = Synapse(pre_id=1, post_id=2, weight=10.0, distance=5.0)
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    spikes = network.run(n_steps=50)

    assert spikes == []


def test_manual_kick_on_cell_a_causes_eventual_spike_on_cell_b() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)  # guarantees cell_a spikes on the first tick

    # Short delay, negligible attenuation, weight big enough to cross
    # cell_b's threshold (-75 -> -50 needs a jump of 25) in one delivery.
    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    spikes = network.run(n_steps=10)

    assert any(spike.neuron_id == cell_b.neuron_id for spike in spikes)


def test_propagation_delay_is_respected() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)  # guarantees cell_a spikes on the first tick

    # distance=50, propagation_speed=1.0 -> delay=50ms. Spike fires at t=1.0,
    # so arrival_time == 51.0. decay_constant kept large so the delivered
    # weight is small but non-zero (we only need to detect a Vm change).
    synapse = Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=50.0, propagation_speed=1.0,
        decay_constant=1000.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    for _ in range(50):
        network.step()
        assert cell_b.Vm == cell_b.Vrest  # arrival_time (51.0) not reached yet

    network.step()  # 51st tick: window [50, 51) now contains arrival_time
    assert cell_b.Vm != cell_b.Vrest


def test_weak_synapse_does_not_trigger_cell_b() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    # Heavy attenuation: effective_weight is effectively zero regardless of
    # the (large) base weight.
    synapse = Synapse(
        pre_id=1, post_id=2, weight=1000.0, distance=1000.0, propagation_speed=1.0,
        decay_constant=1.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    all_spikes = []
    for _ in range(30):
        cell_a.receive_input(100.0)  # repeatedly drive cell_a to spike
        all_spikes.extend(network.step())

    assert all(spike.neuron_id != cell_b.neuron_id for spike in all_spikes)


def test_cell_b_spikes_are_not_repropagated() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    spikes = network.run(n_steps=20)

    # No crash, and only the two known cells ever appear as spike origins —
    # there is no synapse out of cell_b, so nothing else could be produced.
    known_ids = {cell_a.neuron_id, cell_b.neuron_id}
    assert all(spike.neuron_id in known_ids for spike in spikes)


def test_run_returns_spikes_in_chronological_order() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    spikes = network.run(n_steps=20)
    timestamps = [spike.timestamp for spike in spikes]

    assert len(spikes) > 1
    assert timestamps == sorted(timestamps)


def test_current_time_advances_correctly() -> None:
    cell_a = Cell(neuron_id=1)
    cell_b = Cell(neuron_id=2)
    synapse = Synapse(pre_id=1, post_id=2, weight=10.0, distance=5.0)
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=2.0)

    network.run(n_steps=20)

    assert network.current_time == 40.0


def test_no_stdp_update_before_both_cells_have_spiked() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    # Heavy attenuation: cell_b can never reach threshold, so it never
    # spikes, and last_spike_time stays -inf for the whole run. Weight is
    # kept within [w_min, w_max] (unlike an earlier version of this test
    # that used weight=1000.0) — under the trace-based mechanism,
    # on_pre_spike() now touches (and clamps) weight on every single
    # cell_a spike, not just when a qualifying pair exists, so an
    # out-of-bounds starting weight would get silently clamped on the very
    # first spike regardless of dw, which isn't what this test means to check.
    synapse = Synapse(
        pre_id=1, post_id=2, weight=15.0, distance=1000.0, propagation_speed=1.0,
        decay_constant=1.0,
    )
    initial_weight = synapse.weight
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    for _ in range(30):
        cell_a.receive_input(100.0)  # cell_a spikes repeatedly
        network.step()

    # cell_b never spikes -> post_trace stays exactly 0.0 forever -> every
    # on_pre_spike() call computes dw = lr * modulation * A_minus * 0.0 = 0.
    assert synapse.weight == initial_weight


def test_stdp_engages_after_both_cells_have_spiked_once() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)  # guarantees cell_a spikes on the first tick

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0,
    )
    initial_weight = synapse.weight
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    network.run(n_steps=20)

    assert network.synapse.weight != initial_weight


def test_stale_spike_does_not_get_paired_after_fix() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    # Heavy attenuation so propagation can never itself spike cell_b —
    # cell_b's later spike is driven purely by our own direct kick, unrelated
    # to cell_a's long-past spike.
    synapse = Synapse(
        pre_id=1, post_id=2, weight=15.0, distance=1000.0, decay_constant=1.0,
        tau_stdp=20.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    cell_a.receive_input(100.0)
    network.step()  # cell_a's first-ever spike; cell_b's history is empty, no pairing yet
    weight_after_first_spike = synapse.weight

    # Gap far beyond 3 * tau_stdp (60ms).
    for _ in range(200):
        network.step()

    cell_b.receive_input(100.0)  # spikes for a reason wholly unrelated to cell_a
    network.step()

    # Under the OLD scan-based scheme, a hard 3*tau_stdp cutoff fully
    # EXCLUDED anything this stale — an exact equality held. The trace-based
    # scheme has no hard cutoff by design (Song/Miller/Abbott traces only
    # ever decay continuously, never reset or truncate), so pre_trace is not
    # exactly 0 after a ~200ms gap — merely exp(-200/20) ≈ 4.5e-5 of its
    # original value, i.e. negligible rather than excluded. The resulting
    # weight nudge is correspondingly tiny (~4e-7 here), not exactly zero.
    assert abs(synapse.weight - weight_after_first_spike) < 1e-5


def test_multiple_recent_spikes_all_get_paired() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    # Heavy attenuation isolates this test to the deliberate direct kicks
    # below — propagated input can never itself perturb cell_b's timing.
    synapse = Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=1000.0, decay_constant=1.0,
        learning_rate=0.01, tau_stdp=20.0, A_plus=1.0, A_minus=1.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    for _ in range(4):
        cell_a.receive_input(100.0)
        network.step()

    assert cell_a.spike_history == [1.0, 4.0]  # two quick, real causal spikes

    cell_b.receive_input(100.0)
    network.step()  # cell_b's first-ever spike, at t=5.0, triggers one on_post_spike() call

    # Both cell_a spikes (t=1, t=4) must contribute to this single update —
    # not just the nearest one — via pre_trace, which by t=5.0 has
    # accumulated exp(-(5-1)/tau_stdp) + exp(-(5-4)/tau_stdp) (the same
    # exact identity verified directly in
    # test_trace_equivalence_to_multi_pair_sum_analytically). Because this
    # is a SINGLE on_post_spike() call (not two separate update_weight() +
    # record_observation() calls looping over history, as under the old
    # scan-based scheme), there's no prior expectation yet to modulate
    # against — modulation is neutral (1.0) here, and this trace-based path
    # naturally avoids the old scheme's intra-tick contamination (where the
    # first pair's record_observation() call would corrupt the second
    # pair's modulation within the same tick).
    pre_trace_at_t5 = math.exp(-4.0 / 20.0) + math.exp(-1.0 / 20.0)
    expected_dw = 0.01 * 1.0 * 1.0 * pre_trace_at_t5  # learning_rate * modulation * A_plus * pre_trace
    expected = 5.0 + expected_dw
    assert abs(synapse.weight - expected) < 1e-9


def test_current_weight_and_network_still_function_end_to_end() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    # Short delay: cell_b, when it spikes at all, reliably spikes one tick
    # after cell_a (genuine causal timing driven by propagation). w_max is
    # raised so the starting weight (needed for cell_b to reach threshold
    # in one delivery) isn't immediately clamped down by the default 20.0
    # ceiling on the very first update.
    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    initial_weight = synapse.weight
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    # Space trials well beyond 3 * tau_stdp (60ms) so each cell_a kick pairs
    # cleanly against only its own trial's cell_b response — under the fixed
    # multi-pairing scheme, the prior trial's spikes are correctly *excluded*
    # from pairing entirely (not just decayed to near-zero as before the fix).
    for _ in range(3):
        cell_a.receive_input(100.0)
        for _ in range(100):
            network.step()

    assert network.current_weight > initial_weight
    assert network.current_weight == network.synapse.weight


def test_current_weight_property_matches_synapse_weight() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    network.run(n_steps=20)

    assert network.current_weight == network.synapse.weight


def test_stdp_does_not_crash_with_simultaneous_spikes_same_tick() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)
    cell_b.receive_input(100.0)  # force cell_b to also cross threshold this tick

    synapse = Synapse(pre_id=1, post_id=2, weight=10.0, distance=1.0)
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    spikes = network.step()

    assert len(spikes) == 2
    assert {spike.neuron_id for spike in spikes} == {cell_a.neuron_id, cell_b.neuron_id}


def test_record_observation_is_called_during_live_stepping() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    network.run(n_steps=20)

    assert len(synapse.observed_delays) > 0


def test_future_expectation_develops_over_live_run() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    # Repeated, well-spaced causal trials: cell_b reliably spikes ~1 tick
    # after cell_a (propagation delay=1ms), so the typical observed delay
    # should converge to roughly 1.0.
    for _ in range(3):
        cell_a.receive_input(100.0)
        for _ in range(100):
            network.step()

    assert synapse.future_expectation is not None
    assert abs(synapse.future_expectation - 1.0) < 0.5


def test_modulation_affects_live_weight_updates_over_time() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    weight_changes = []
    for _ in range(4):
        weight_before = synapse.weight
        cell_a.receive_input(100.0)
        for _ in range(100):
            network.step()
        weight_changes.append(synapse.weight - weight_before)

    # First trial: no prior expectation yet -> modulation=1.0 (unmodulated).
    # Later trials: the same ~1ms delay keeps getting confirmed, so
    # weighted_error shrinks toward 0 and modulation shrinks toward m_min ->
    # each successive update should move the weight less, not more.
    assert weight_changes[-1] < weight_changes[0]


def test_call_order_produces_correct_first_pair_behavior() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)
    initial_weight = synapse.weight

    # Enough steps for cell_a to spike (t=1.0) and, via the 1ms-delay
    # synapse, cell_b to spike shortly after (t=2.0) — the network's only
    # qualifying pair within this short run.
    network.run(n_steps=5)

    # If update_weight() had run AFTER record_observation() for this same
    # pair, future_expectation would already equal this pair's own delay,
    # making weighted_error (and thus the modulation deviation from 1.0)
    # artificially collapse to 0 regardless of correctness. Asserting the
    # weight change matches the fully UNMODULATED formula confirms
    # update_weight() genuinely ran first, against a still-None expectation.
    expected_dw = synapse.learning_rate * synapse.A_plus * math.exp(-1.0 / synapse.tau_stdp)
    assert abs(synapse.weight - (initial_weight + expected_dw)) < 1e-9


def test_live_network_uses_trace_based_stdp_now() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    network.run(n_steps=20)

    # Confirms the O(1) trace path, not the old history scan, is what's
    # actually driving live weight updates now.
    assert synapse.pre_trace > 0.0 or synapse.post_trace > 0.0
    # And the prediction-machinery feed (record_observation via
    # on_post_spike) still works end-to-end through the new path.
    assert len(synapse.observed_delays) > 0


def test_causal_pattern_still_potentiates_with_traces() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    initial_weight = synapse.weight
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    # Repeated, well-spaced causal trials: cell_a reliably precedes cell_b
    # by ~1ms each time. The corrected version of the earlier
    # causal-ordering tests, now validated under the trace-based mechanism.
    for _ in range(3):
        cell_a.receive_input(100.0)
        for _ in range(100):
            network.step()

    assert network.current_weight > initial_weight


def test_live_network_populates_trace_context_with_synapse_source() -> None:
    cell_a = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_b = Cell(neuron_id=2, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell_a.receive_input(100.0)  # guarantees cell_a spikes on the first tick

    synapse = Synapse(
        pre_id=1, post_id=2, weight=30.0, distance=1.0, propagation_speed=1.0,
        decay_constant=10.0, w_max=50.0,
    )
    network = TwoCellNetwork(cell_a, cell_b, synapse, dt=1.0)

    network.run(n_steps=10)  # enough for propagation delivery to reach cell_b

    assert cell_b.trace_context is not None
    source_id, _ = cell_b.trace_context
    assert source_id == synapse.pre_id == cell_a.neuron_id
