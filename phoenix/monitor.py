"""Passive activity monitoring for a Phoenix ``Cell``.

``ActivityMonitor`` is a diagnostic observer, not a learning mechanism: it
tracks the spike timestamps a caller feeds it and reports firing rate,
silence, and saturation over a sliding time window. This exists so a cell's
dynamics can be verified as stable — neither dead nor exploding — before any
homeostasis or STDP mechanism is built on top of it. It never touches a
``Cell`` object directly.
"""

from __future__ import annotations

from phoenix.spike import Spike

_MS_PER_SECOND = 1000.0


class ActivityMonitor:
    """Tracks spike timestamps for one cell over a sliding time window."""

    def __init__(self, window_size: float) -> None:
        self.window_size: float = window_size
        self._spike_times: list[float] = []

    def record_spike(self, spike: Spike) -> None:
        """Record a spike's timestamp. Purely observational, no side effects on the cell."""
        self._spike_times.append(spike.timestamp)

    def firing_rate(self, current_time: float) -> float:
        """Spikes per second within [current_time - window_size, current_time]."""
        lower_bound = current_time - self.window_size
        count = sum(
            1 for ts in self._spike_times if lower_bound <= ts <= current_time
        )
        window_seconds = self.window_size / _MS_PER_SECOND
        return count / window_seconds

    def is_silent(self, current_time: float, silence_threshold_ms: float) -> bool:
        """True if no spike has occurred within the last silence_threshold_ms."""
        if not self._spike_times:
            return True
        elapsed = current_time - max(self._spike_times)
        return elapsed > silence_threshold_ms

    def is_saturated(self, current_time: float, rate_threshold_hz: float) -> bool:
        """True if the current firing rate exceeds rate_threshold_hz."""
        return self.firing_rate(current_time) > rate_threshold_hz
