"""Single-cell closed-loop driver for Phoenix.

``CellRunner`` closes the fundamental feedback loop that the unit tests leave
open:

    integrate -> real spike -> monitor -> apply_homeostasis -> Vthresh -> next integrate

The contract this driver enforces (and which the unit tests deliberately do
*not*):

- Every spike that reaches the monitor is the genuine return value of
  ``cell.integrate(dt)`` — never a hand-inserted ``Spike``.
- The clock advances *only* through ``integrate`` — ``cell.t`` is never
  poked directly.
- External input reaches the cell *only* through ``receive_input`` — ``Vm``
  is never assigned to force a spike.

This is a driver only. It holds no membrane/threshold/spike dynamics of its
own; all of that lives in ``Cell``. Its sole jobs are ordering the per-step
calls and logging time series for inspection.
"""

from __future__ import annotations

from collections.abc import Callable

from phoenix.cell import Cell
from phoenix.monitor import ActivityMonitor
from phoenix.spike import Spike

# An input source maps the cell's current time (ms) to a synaptic weight to
# inject this step (0.0 / falsy means "inject nothing this step").
InputSource = Callable[[float], float]


class CellRunner:
    """Drives ONE ``Cell`` through a fixed-``dt`` closed feedback loop."""

    def __init__(
        self,
        cell: Cell,
        monitor: ActivityMonitor,
        dt: float = 1.0,
        input_source: InputSource | None = None,
    ) -> None:
        self.cell: Cell = cell
        self.monitor: ActivityMonitor = monitor  # a REAL ActivityMonitor, never a stub
        self.dt: float = dt
        self.input_source: InputSource | None = input_source

        # Time-series logs for inspection (one entry per step).
        self.times: list[float] = []
        self.Vm: list[float] = []
        self.Vthresh: list[float] = []
        self.rate: list[float] = []
        # Genuine spike timestamps (from integrate), for independent rate math.
        self.spike_times: list[float] = []

    def step(self) -> Spike | None:
        """Advance the closed loop by exactly one tick, in the fixed order."""
        # (0) Inject any external input BEFORE the step, via the synaptic
        #     pathway only. During refractory the cell drops it itself.
        if self.input_source is not None:
            weight = self.input_source(self.cell.t)
            if weight:
                self.cell.receive_input(weight)

        # (1) Advance the cell. `spike` is the REAL integrate output.
        spike = self.cell.integrate(self.dt)

        # (2) Feed only that genuine spike to the monitor.
        if spike is not None:
            self.monitor.record_spike(spike)
            self.spike_times.append(spike.timestamp)

        # (3) Homeostatic threshold adaptation from observed activity.
        self.cell.apply_homeostasis(self.monitor, self.dt)

        # (4) Spontaneous sub-threshold activity if the cell has gone silent.
        self.cell.maybe_spontaneous_activity(self.monitor)

        # Log after the fully-applied step.
        self.times.append(self.cell.t)
        self.Vm.append(self.cell.Vm)
        self.Vthresh.append(self.cell.Vthresh)
        self.rate.append(self.monitor.firing_rate(self.cell.t))
        return spike

    def run(self, n_steps: int) -> None:
        """Run a fixed number of steps."""
        for _ in range(n_steps):
            self.step()

    def run_for(self, duration_ms: float) -> None:
        """Run for approximately ``duration_ms`` of simulated time."""
        n_steps = int(round(duration_ms / self.dt))
        for _ in range(n_steps):
            self.step()

    # --- inspection helpers (pure reads over the logged series) ---

    def window_rate(self, t_start: float, t_end: float) -> float:
        """True firing rate (Hz) from genuine spikes in [t_start, t_end]."""
        count = sum(1 for ts in self.spike_times if t_start <= ts <= t_end)
        seconds = (t_end - t_start) / 1000.0
        return count / seconds if seconds > 0 else 0.0

    def tail_indices(self, fraction: float) -> range:
        """Indices of the final ``fraction`` of logged steps."""
        n = len(self.times)
        start = int(n * (1.0 - fraction))
        return range(start, n)

    def tail_vthresh(self, fraction: float) -> list[float]:
        """Logged ``Vthresh`` values over the final ``fraction`` of the run."""
        return [self.Vthresh[i] for i in self.tail_indices(fraction)]
