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


def test_is_silent_survives_threshold_longer_than_window() -> None:
    """Regression [N1-b]: bounded memory must NOT break silence detection.

    silence_threshold_ms is passed per call and may EXCEED window_size, so
    is_silent cannot read the pruned rate window — it must read the separate,
    never-pruned last-spike time. The naive fix (prune everything older than
    window_size, then answer is_silent from what's left) fails exactly here:
    the spike at t=50 falls out of the 100 ms rate window by t=180, and a
    monitor that forgot it would wrongly call this cell silent — reporting a
    live cell as dying.
    """
    monitor = ActivityMonitor(window_size=100.0)
    monitor.record_spike(_spike_at(50.0))

    # Only 130 ms since the spike, against a 200 ms silence threshold.
    assert monitor.is_silent(current_time=180.0, silence_threshold_ms=200.0) is False

    # The spike is genuinely outside the 100 ms RATE window (rate is 0)...
    assert monitor.firing_rate(current_time=180.0) == 0.0
    # ...yet silence detection still sees it. Rate memory is bounded; the
    # last-spike time is not.
    assert monitor.is_silent(current_time=180.0, silence_threshold_ms=200.0) is False

    # And once the threshold really is exceeded, it does report silence.
    assert monitor.is_silent(current_time=300.0, silence_threshold_ms=200.0) is True


def test_is_saturated_true_above_rate_threshold() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    for ts in range(0, 100, 10):  # 10 spikes within a 100ms window -> 100 Hz
        monitor.record_spike(_spike_at(ts))

    assert monitor.is_saturated(current_time=100.0, rate_threshold_hz=50.0) is True


def test_is_saturated_false_below_rate_threshold() -> None:
    monitor = ActivityMonitor(window_size=100.0)
    monitor.record_spike(_spike_at(10.0))

    assert monitor.is_saturated(current_time=100.0, rate_threshold_hz=50.0) is False
