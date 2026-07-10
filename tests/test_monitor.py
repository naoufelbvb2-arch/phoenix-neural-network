"""Tests for the Phoenix ``ActivityMonitor`` diagnostic layer."""

from phoenix.monitor import ActivityMonitor
from phoenix.spike import Spike


def _spike_at(timestamp: float) -> Spike:
    return Spike(neuron_id=1, timestamp=timestamp)


def test_empty_monitor_is_silent() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    assert monitor.is_silent(current_time=0.0, silence_threshold_ms=10.0) is True


def test_firing_rate_zero_with_no_spikes() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    assert monitor.firing_rate(current_time=100.0) == 0.0


def test_firing_rate_correct_count_in_window() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    for ts in (10, 20, 30, 90, 150):
        monitor.record_spike(_spike_at(ts))

    assert monitor.firing_rate(current_time=100.0) == 40.0


def test_firing_rate_excludes_out_of_window_spikes() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    for ts in (10, 20, 30, 90, 150):
        monitor.record_spike(_spike_at(ts))

    assert monitor.firing_rate(current_time=200.0) == 10.0


def test_is_silent_true_after_long_gap() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    monitor.record_spike(_spike_at(10.0))

    assert monitor.is_silent(current_time=500.0, silence_threshold_ms=50.0) is True


def test_is_silent_false_with_recent_spike() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    monitor.record_spike(_spike_at(10.0))

    assert monitor.is_silent(current_time=15.0, silence_threshold_ms=50.0) is False


def test_is_saturated_true_above_rate_threshold() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    for ts in range(0, 100, 10):  # 10 spikes within a 100ms window -> 100 Hz
        monitor.record_spike(_spike_at(ts))

    assert monitor.is_saturated(current_time=100.0, rate_threshold_hz=50.0) is True


def test_is_saturated_false_below_rate_threshold() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    monitor.record_spike(_spike_at(10.0))

    assert monitor.is_saturated(current_time=100.0, rate_threshold_hz=50.0) is False
