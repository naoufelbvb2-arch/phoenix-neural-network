"""Tests for the Phoenix ``Cell`` state and passive leak dynamics."""

import math

from phoenix.cell import Cell
from phoenix.monitor import ActivityMonitor
from phoenix.spike import Spike


def test_initial_state() -> None:
    cell = Cell(neuron_id=1)
    assert cell.Vm == cell.Vrest
    assert cell.last_spike_time == -math.inf
    assert cell.t == 0.0


def test_leak_exact_formula_single_step() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, tau=20.0)
    cell.Vm = -50.0
    Vm0 = cell.Vm

    cell.leak(dt=10.0)

    expected = -75.0 + (Vm0 - (-75.0)) * math.exp(-10.0 / 20.0)
    assert abs(cell.Vm - expected) < 1e-9


def test_leak_matches_analytical_at_multiple_timepoints() -> None:
    Vrest, tau, Vm0 = -75.0, 20.0, -50.0
    cell = Cell(neuron_id=1, Vrest=Vrest, tau=tau)
    cell.Vm = Vm0

    checkpoints = {5, 15, 30}
    for step in range(1, 31):
        cell.leak(dt=1.0)
        if step in checkpoints:
            expected = Vrest + (Vm0 - Vrest) * math.exp(-step / tau)
            assert abs(cell.Vm - expected) < 1e-6


def test_leak_is_dt_invariant() -> None:
    cell_a = Cell(neuron_id=1)
    cell_a.Vm = -50.0
    cell_a.leak(dt=10.0)

    cell_b = Cell(neuron_id=1)
    cell_b.Vm = -50.0
    for _ in range(10):
        cell_b.leak(dt=1.0)

    assert abs(cell_a.Vm - cell_b.Vm) < 1e-9


def test_leak_from_rest_stays_at_rest() -> None:
    cell = Cell(neuron_id=1)
    assert cell.Vm == cell.Vrest

    cell.leak(dt=5.0)

    assert abs(cell.Vm - cell.Vrest) < 1e-9


def test_clock_advances_correctly() -> None:
    cell = Cell(neuron_id=1)
    cell.leak(dt=5.0)
    cell.leak(dt=5.0)
    cell.leak(dt=5.0)
    assert cell.t == 15.0


def test_large_dt_does_not_overshoot() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, tau=20.0)
    cell.Vm = -50.0

    cell.leak(dt=1000.0)

    # Decays essentially to rest, and never crosses past it.
    assert abs(cell.Vm - cell.Vrest) < 0.01
    assert cell.Vm >= cell.Vrest


def test_receive_input_accumulates() -> None:
    cell = Cell(neuron_id=1)
    cell.receive_input(2.0)
    cell.receive_input(2.0)
    assert cell.input_current == 4.0


def test_integrate_applies_leak_then_input() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(10.0)
    cell.integrate(dt=0.0)

    assert cell.Vm == -65.0


def test_integrate_resets_input_current() -> None:
    cell = Cell(neuron_id=1)
    cell.receive_input(5.0)
    cell.integrate(dt=1.0)
    assert cell.input_current == 0.0


def test_integrate_combines_leak_and_input_correctly() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, tau=20.0)
    cell.Vm = -60.0

    cell.receive_input(3.0)
    cell.integrate(dt=10.0)

    leaked_Vm = -75.0 + (-60.0 - (-75.0)) * math.exp(-10.0 / 20.0)
    expected = leaked_Vm + 3.0
    assert abs(cell.Vm - expected) < 1e-9


def test_integrate_advances_clock() -> None:
    cell = Cell(neuron_id=1)
    cell.integrate(dt=7.0)
    assert cell.t == 7.0


def test_multiple_integrate_steps_accumulate_time_correctly() -> None:
    cell_integrate = Cell(neuron_id=1)
    cell_integrate.Vm = -50.0
    for _ in range(3):
        cell_integrate.integrate(dt=5.0)

    cell_leak = Cell(neuron_id=1)
    cell_leak.Vm = -50.0
    for _ in range(3):
        cell_leak.leak(dt=5.0)

    assert abs(cell_integrate.Vm - cell_leak.Vm) < 1e-9
    assert cell_integrate.t == cell_leak.t


def test_no_spike_below_threshold() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(1.0)
    result = cell.integrate(dt=1.0)

    assert result is None


def test_spike_fires_at_threshold() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(25.0)
    result = cell.integrate(dt=0.0)

    assert isinstance(result, Spike)


def test_spike_resets_vm() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, Vreset=-75.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(25.0)
    cell.integrate(dt=0.0)

    assert cell.Vm == cell.Vreset


def test_spike_records_last_spike_time() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(25.0)
    cell.integrate(dt=3.0)

    assert cell.last_spike_time == cell.t


def test_spike_object_fields() -> None:
    cell = Cell(neuron_id=7, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(25.0)
    result = cell.integrate(dt=5.0)

    assert isinstance(result, Spike)
    assert result.neuron_id == 7
    assert result.timestamp == cell.t
    assert result.amplitude == 40.0


def test_spike_above_threshold_also_fires() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(50.0)
    result = cell.integrate(dt=0.0)

    assert isinstance(result, Spike)


def test_no_spike_returns_none_explicitly() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -75.0

    cell.receive_input(1.0)
    result = cell.integrate(dt=1.0)

    assert result is None


def test_input_rejected_during_refractory() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, refractory_period=2.0)
    cell.t = 5.0
    cell.refractory_until = 5.0 + cell.refractory_period

    cell.receive_input(10.0)

    assert cell.input_current == 0.0


def test_input_accepted_after_refractory_expires() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, refractory_period=2.0)
    cell.refractory_until = 2.0

    cell.integrate(dt=2.0)  # advances t to 2.0, refractory has just expired
    cell.receive_input(5.0)

    assert cell.input_current == 5.0


def test_no_second_spike_during_refractory_even_if_vm_high() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -40.0  # above Vthresh
    cell.refractory_until = 10.0  # still refractory at t == 0.0

    result = cell.integrate(dt=0.0)

    assert result is None


def test_spike_sets_refractory_until_correctly() -> None:
    cell = Cell(
        neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0, refractory_period=2.0
    )
    cell.Vm = -75.0

    cell.receive_input(25.0)
    cell.integrate(dt=3.0)

    assert cell.refractory_until == cell.t + cell.refractory_period


def test_refractory_boundary_is_inclusive_correctly() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0)
    cell.Vm = -40.0  # above Vthresh
    cell.refractory_until = 5.0
    cell.t = 5.0  # exactly at the boundary: refractory has ended

    result = cell.integrate(dt=0.0)

    assert isinstance(result, Spike)


def test_leak_still_applies_during_refractory() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -60.0
    cell.refractory_until = 100.0  # deep in refractory

    cell.integrate(dt=10.0)

    expected = -75.0 + (-60.0 - (-75.0)) * math.exp(-10.0 / 20.0)
    assert abs(cell.Vm - expected) < 1e-9


def test_full_refractory_cycle_end_to_end() -> None:
    cell = Cell(
        neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0, refractory_period=5.0
    )
    cell.Vm = -75.0

    # Drive the cell to spike.
    cell.receive_input(25.0)
    first_spike = cell.integrate(dt=0.0)
    assert isinstance(first_spike, Spike)
    assert cell.refractory_until == 5.0

    # Try to spike again immediately: refractory should suppress it, even
    # with strong input (input is dropped, so this also proves rejection).
    cell.receive_input(50.0)
    result = cell.integrate(dt=1.0)
    assert result is None

    cell.receive_input(50.0)
    result = cell.integrate(dt=1.0)
    assert result is None

    # Advance past the refractory window, then drive Vm back up.
    result = cell.integrate(dt=10.0)
    assert result is None
    assert cell.t >= cell.refractory_until

    cell.receive_input(25.0)
    second_spike = cell.integrate(dt=0.0)
    assert isinstance(second_spike, Spike)


def _overactive_monitor(current_time: float) -> ActivityMonitor:
    # 5 spikes packed into the trailing 100ms window -> 50 Hz, well above
    # target_rate_hz=5.0. Anchored relative to current_time so this stays
    # valid even as the caller advances the clock across repeated calls.
    monitor = ActivityMonitor(window_size=100.0)
    window_start = current_time - 100.0
    for offset in (0.0, 20.0, 40.0, 60.0, 80.0):
        monitor.record_spike(Spike(neuron_id=1, timestamp=window_start + offset))
    assert monitor.firing_rate(current_time) == 50.0
    return monitor


def test_homeostasis_raises_threshold_when_overactive() -> None:
    cell = Cell(neuron_id=1, tau_homeostasis=10000.0, target_rate_hz=5.0)
    cell.t = 100.0
    initial_Vthresh = cell.Vthresh

    monitor = _overactive_monitor(cell.t)
    cell.apply_homeostasis(monitor, dt=100.0)

    assert cell.Vthresh > initial_Vthresh


def test_homeostasis_lowers_threshold_when_underactive() -> None:
    cell = Cell(neuron_id=1, tau_homeostasis=10000.0, target_rate_hz=5.0)
    cell.t = 100.0
    initial_Vthresh = cell.Vthresh

    monitor = ActivityMonitor(window_size=100.0)  # no spikes recorded
    cell.apply_homeostasis(monitor, dt=100.0)

    assert cell.Vthresh < initial_Vthresh


def test_homeostasis_change_is_small_per_call() -> None:
    cell = Cell(neuron_id=1, tau_homeostasis=10000.0, target_rate_hz=5.0)
    cell.t = 100.0
    initial_Vthresh = cell.Vthresh

    monitor = _overactive_monitor(cell.t)
    cell.apply_homeostasis(monitor, dt=10.0)

    assert abs(cell.Vthresh - initial_Vthresh) < 0.1


def test_homeostasis_respects_upper_bound() -> None:
    cell = Cell(
        neuron_id=1, tau_homeostasis=10000.0, target_rate_hz=5.0, Vthresh_max=-40.0
    )

    for _ in range(200):
        cell.t += 100.0
        monitor = _overactive_monitor(cell.t)
        cell.apply_homeostasis(monitor, dt=100.0)
        assert cell.Vthresh <= cell.Vthresh_max

    assert cell.Vthresh == cell.Vthresh_max


def test_homeostasis_respects_lower_bound() -> None:
    cell = Cell(
        neuron_id=1, tau_homeostasis=10000.0, target_rate_hz=5.0, Vthresh_min=-55.0
    )
    silent_monitor = ActivityMonitor(window_size=100.0)  # no spikes recorded

    for _ in range(200):
        cell.t += 100.0
        cell.apply_homeostasis(silent_monitor, dt=100.0)
        assert cell.Vthresh >= cell.Vthresh_min

    assert cell.Vthresh == cell.Vthresh_min


def test_homeostasis_no_change_at_exact_target_rate() -> None:
    cell = Cell(neuron_id=1, tau_homeostasis=10000.0, target_rate_hz=5.0)
    cell.t = 500.0
    initial_Vthresh = cell.Vthresh

    # 5 spikes within a 1000ms window -> exactly 5.0 Hz, matching target.
    monitor = ActivityMonitor(window_size=1000.0)
    for ts in (100.0, 200.0, 300.0, 400.0, 500.0):
        monitor.record_spike(Spike(neuron_id=1, timestamp=ts))
    assert monitor.firing_rate(cell.t) == 5.0

    cell.apply_homeostasis(monitor, dt=10.0)

    assert abs(cell.Vthresh - initial_Vthresh) < 1e-9


def test_homeostasis_does_not_run_automatically_in_integrate() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    initial_Vthresh = cell.Vthresh

    cell.receive_input(5.0)
    cell.integrate(dt=1.0)
    cell.receive_input(5.0)
    cell.integrate(dt=1.0)
    cell.receive_input(5.0)
    cell.integrate(dt=1.0)

    assert cell.Vthresh == initial_Vthresh


def test_spike_history_records_timestamps() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0, refractory_period=2.0)

    timestamps = []
    for _ in range(3):
        cell.integrate(dt=2.0)  # let any refractory clear first, no input -> no spike
        cell.receive_input(25.0)
        spike = cell.integrate(dt=0.0)
        assert spike is not None
        timestamps.append(spike.timestamp)

    assert timestamps == [2.0, 4.0, 6.0]
    assert cell.spike_history == timestamps


def test_spike_history_prunes_old_entries() -> None:
    cell = Cell(
        neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0, refractory_period=2.0,
        stdp_history_window=10.0,
    )

    timestamps = []
    for _ in range(7):
        cell.integrate(dt=2.0)
        cell.receive_input(25.0)
        spike = cell.integrate(dt=0.0)
        assert spike is not None
        timestamps.append(spike.timestamp)

    # Spikes land at t = 2, 4, ..., 14. By the last spike (t=14), the
    # window (10.0) only retains entries with ts >= 14 - 10 = 4.
    assert timestamps[0] == 2.0
    assert timestamps[0] not in cell.spike_history
    assert timestamps[-1] in cell.spike_history


def test_trace_context_none_initially() -> None:
    cell = Cell(neuron_id=1)
    assert cell.trace_context is None


def test_trace_context_updates_on_accepted_input() -> None:
    cell = Cell(neuron_id=1)

    cell.receive_input(weight=5.0, source_id=42)

    assert cell.trace_context == (42, cell.t)


def test_trace_context_reflects_most_recent_source_only() -> None:
    cell = Cell(neuron_id=1)

    cell.receive_input(weight=2.0, source_id=1)
    cell.receive_input(weight=3.0, source_id=2)

    assert cell.trace_context == (2, cell.t)


def test_trace_context_unaffected_by_rejected_input_during_refractory() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, refractory_period=2.0)
    cell.receive_input(weight=25.0, source_id=7)
    spike = cell.integrate(dt=0.0)
    assert spike is not None  # forced a spike, setting refractory_until forward

    trace_context_before = cell.trace_context
    assert cell.t < cell.refractory_until  # still refractory

    cell.receive_input(weight=5.0, source_id=99)

    assert cell.trace_context == trace_context_before


def test_trace_context_source_id_optional_backward_compatible() -> None:
    cell = Cell(neuron_id=1)

    cell.receive_input(weight=5.0)  # no source_id, old-style call

    assert cell.input_current == 5.0
    assert cell.trace_context == (None, cell.t)


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def test_no_spontaneous_activity_when_not_silent() -> None:
    cell = Cell(neuron_id=1, rng_seed=1)
    monitor = ActivityMonitor(window_size=100.0)
    monitor.record_spike(Spike(neuron_id=1, timestamp=cell.t))  # spike "now" -> not silent

    initial_Vm = cell.Vm
    result = cell.maybe_spontaneous_activity(monitor)

    assert result is False
    assert cell.Vm == initial_Vm


def test_spontaneous_activity_triggers_when_silent() -> None:
    cell = Cell(neuron_id=1, rng_seed=42)
    monitor = ActivityMonitor(window_size=100.0)  # no spikes recorded -> is_silent

    initial_Vm = cell.Vm
    result = cell.maybe_spontaneous_activity(monitor)

    assert result is True
    assert cell.Vm != initial_Vm


def test_spontaneous_noise_is_never_by_itself_above_threshold() -> None:
    cell = Cell(
        neuron_id=1, Vthresh=-50.0, spontaneous_noise_std=10.0, rng_seed=1
    )
    cell.Vm = -50.5  # already very close to threshold
    monitor = ActivityMonitor(window_size=100.0)  # empty -> always silent

    for _ in range(100):
        cell.maybe_spontaneous_activity(monitor)
        assert cell.Vm < cell.Vthresh


def test_spontaneous_activity_is_deterministic_with_seed() -> None:
    cell_a = Cell(neuron_id=1, rng_seed=123)
    cell_b = Cell(neuron_id=2, rng_seed=123)
    monitor_a = ActivityMonitor(window_size=100.0)
    monitor_b = ActivityMonitor(window_size=100.0)

    cell_a.maybe_spontaneous_activity(monitor_a)
    cell_b.maybe_spontaneous_activity(monitor_b)

    assert cell_a.Vm == cell_b.Vm


def test_spontaneous_activity_not_auto_triggered_by_integrate() -> None:
    cell = Cell(neuron_id=1, Vrest=-75.0, Vthresh=-50.0, tau=20.0)
    cell.Vm = -60.0

    for _ in range(5):
        cell.integrate(dt=1.0)

    expected = -75.0 + (-60.0 - (-75.0)) * math.exp(-5.0 / 20.0)
    assert abs(cell.Vm - expected) < 1e-9


def test_spontaneous_noise_std_affects_magnitude() -> None:
    cell_low = Cell(neuron_id=1, spontaneous_noise_std=0.1, rng_seed=7)
    cell_high = Cell(neuron_id=2, spontaneous_noise_std=10.0, rng_seed=7)
    monitor = ActivityMonitor(window_size=100.0)  # empty -> always silent

    low_values = []
    high_values = []
    for _ in range(50):
        cell_low.Vm = cell_low.Vrest
        cell_low.maybe_spontaneous_activity(monitor)
        low_values.append(cell_low.Vm)

        cell_high.Vm = cell_high.Vrest
        cell_high.maybe_spontaneous_activity(monitor)
        high_values.append(cell_high.Vm)

    assert _variance(high_values) > _variance(low_values)
