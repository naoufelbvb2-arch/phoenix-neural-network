"""Passive activity monitoring for a Phoenix ``Cell``.

``ActivityMonitor`` is a diagnostic observer, not a learning mechanism: it
tracks the spike timestamps a caller feeds it and reports firing rate,
silence, and saturation over a sliding time window. This exists so a cell's
dynamics can be verified as stable — neither dead nor exploding — before any
homeostasis or STDP mechanism is built on top of it. It never touches a
``Cell`` object directly.

Memory is BOUNDED: only spikes still inside the rate window are retained, so
storage is O(spikes-in-window) rather than O(total-spikes-ever) and
``firing_rate`` is O(1) instead of rescanning all history.

>>> Why a second, never-pruned field is required <<<
Pruning everything older than ``window_size`` and letting ``is_silent`` read
the pruned structure is WRONG. ``silence_threshold_ms`` is passed per call and
may be LONGER than ``window_size``. Concretely: ``window_size=100``, one spike
at t=50, now t=180 — the correct answer to ``is_silent(180, 200)`` is False
(only 130 ms since the spike), but a naively-pruned monitor has forgotten that
spike and would wrongly answer True (i.e. report a live cell as dying). So
``_last_spike_time`` is kept separately and NEVER pruned, and ``is_silent``
reads only that.
"""

from __future__ import annotations

from collections import deque

from phoenix.spike import Spike

_MS_PER_SECOND = 1000.0


class ActivityMonitor:
    """Tracks spike timestamps for one cell over a sliding time window."""

    def __init__(self, window_size: float) -> None:
        self.window_size: float = window_size
        # Bounded: holds only the spikes currently inside the rate window.
        self._spike_times: deque[float] = deque()
        # NEVER pruned — see the module docstring. is_silent depends on this.
        self._last_spike_time: float | None = None

    def record_spike(self, spike: Spike) -> None:
        """Record a spike's timestamp. Purely observational, no side effects on the cell."""
        self._spike_times.append(spike.timestamp)
        if self._last_spike_time is None or spike.timestamp > self._last_spike_time:
            self._last_spike_time = spike.timestamp

    def _prune(self, current_time: float) -> None:
        """Drop spikes that have fallen out of the window trailing ``current_time``.

        Pruning is done LAZILY, at query time against the QUERY's own
        ``current_time`` — not eagerly inside ``record_spike`` against the
        newest spike's timestamp. Eager pruning would be destructive: recording
        a spike at t=150 (window 100) would discard the spikes at t=10/20/30,
        after which a query about t=100 can no longer be answered correctly.
        Lazy pruning keeps memory bounded all the same, because every live path
        (``apply_homeostasis``, ``CellRunner``) queries ``firing_rate`` on every
        tick, which prunes on every tick.
        """
        lower_bound = current_time - self.window_size
        while self._spike_times and self._spike_times[0] < lower_bound:
            self._spike_times.popleft()

    def firing_rate(self, current_time: float) -> float:
        """Spikes per second within [current_time - window_size, current_time]."""
        self._prune(current_time)

        # Exclude any retained spike stamped LATER than current_time. In live
        # use (monotonic clocks) there are none and this loop exits at once, so
        # the call is O(1); it only does work for out-of-order queries, where
        # the deque's ascending order keeps it cheap.
        ahead = 0
        for timestamp in reversed(self._spike_times):
            if timestamp > current_time:
                ahead += 1
            else:
                break

        count = len(self._spike_times) - ahead
        return count / (self.window_size / _MS_PER_SECOND)

    def is_silent(self, current_time: float, silence_threshold_ms: float) -> bool:
        """True if no spike has occurred within the last silence_threshold_ms.

        Reads ``_last_spike_time`` (never pruned), NOT the windowed deque —
        ``silence_threshold_ms`` may exceed ``window_size``.
        """
        if self._last_spike_time is None:
            return True
        return (current_time - self._last_spike_time) > silence_threshold_ms

    def is_saturated(self, current_time: float, rate_threshold_hz: float) -> bool:
        """True if the current firing rate exceeds rate_threshold_hz."""
        return self.firing_rate(current_time) > rate_threshold_hz
