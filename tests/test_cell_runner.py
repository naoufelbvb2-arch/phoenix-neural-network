"""Closed-loop integration test for a single Phoenix cell.

Unlike the unit tests (which drive each regulatory mechanism against a
hand-fed monitor while faking the other half of the loop), this exercises the
FULL feedback loop through ``CellRunner``:

    integrate -> real spike -> monitor -> apply_homeostasis -> Vthresh -> next integrate

Nothing here inserts spikes into the monitor by hand, pokes ``cell.t``, or
assigns ``cell.Vm`` to force a spike. Input reaches the cell only via
``receive_input``; time advances only via ``integrate``.

Regime choice (documented, not arbitrary): the membrane constant is set to
``tau=100`` ms so the 5 Hz homeostatic setpoint sits ~2 membrane time
constants per inter-spike interval — a healthy supra-threshold regime with an
unambiguous interior equilibrium. At the DEFAULT ``tau=20`` ms the same 5 Hz
setpoint sits ~10 tau per ISI (near the firing cliff / rheobase); the loop
still converges there but ~4x noisier (Vthresh sigma ~0.24 vs ~0.06 mV). See
the task report for that finding. This is a constructor-parameter choice, not
a change to any cell/monitor/homeostasis logic.
"""

from __future__ import annotations

import math
import statistics

import pytest

from phoenix.cell import Cell
from phoenix.monitor import ActivityMonitor
from phoenix.runner import CellRunner

# --- shared regime constants for the headline convergence run ---
TAU = 100.0
TARGET_HZ = 5.0
VTHRESH_INIT = -53.0
VTHRESH_MIN = -55.0
VTHRESH_MAX = -40.0
INPUT_W = 0.31          # constant per-ms drive placing equilibrium interior
WINDOW_MS = 2000.0      # rate window: ~10 spikes at 5 Hz -> 0.5 Hz resolution
SILENCE_THR_MS = 1000.0  # > ISI(=200ms), so spontaneous fires only in real silence
RUN_MS = 60000.0        # 60 s: tens of seconds, so Vthresh moves tangibly
DT = 1.0


def _build_cell(vthresh_init: float = VTHRESH_INIT) -> Cell:
    return Cell(
        neuron_id=1,
        tau=TAU,
        target_rate_hz=TARGET_HZ,
        Vthresh=vthresh_init,
        Vthresh_min=VTHRESH_MIN,
        Vthresh_max=VTHRESH_MAX,
        tau_homeostasis=10000.0,
        refractory_period=2.0,
        spontaneous_silence_threshold_ms=SILENCE_THR_MS,
        spontaneous_noise_std=1.0,
        rng_seed=0,
    )


@pytest.fixture(scope="module")
def converged() -> CellRunner:
    """Run the headline closed-loop convergence ONCE, shared by AC1-AC4."""
    cell = _build_cell()
    monitor = ActivityMonitor(window_size=WINDOW_MS)
    runner = CellRunner(cell, monitor, dt=DT, input_source=lambda t: INPUT_W)
    runner.run_for(RUN_MS)
    return runner


# ---------------------------------------------------------------------------
# [AC1] Actually a closed loop: every monitored spike is a real integrate output
# ---------------------------------------------------------------------------
def test_ac1_loop_is_genuinely_closed(converged: CellRunner) -> None:
    runner = converged

    # The runner records to the monitor ONLY the genuine return value of
    # integrate (see CellRunner.step). So the monitor's spike history must be
    # exactly the runner's recorded integrate outputs — same values, order.
    assert runner.monitor._spike_times == runner.spike_times
    assert len(runner.spike_times) > 0

    # Real spikes: strictly increasing in time, none beyond the simulated horizon.
    assert all(b > a for a, b in zip(runner.spike_times, runner.spike_times[1:]))
    assert max(runner.spike_times) <= RUN_MS

    # Every spike timestamp lands on the cell's own clock grid (came through
    # integrate advancing t by dt), never an out-of-band injected value.
    assert all(abs((ts / DT) - round(ts / DT)) < 1e-9 for ts in runner.spike_times)


# ---------------------------------------------------------------------------
# [AC2] Internal equilibrium: Vthresh settles strictly INSIDE its bounds
# ---------------------------------------------------------------------------
def test_ac2_equilibrium_is_interior_not_pinned(converged: CellRunner) -> None:
    runner = converged
    tail = runner.tail_vthresh(0.25)

    # If homeostasis had merely saturated at a bound, this would be ~Vthresh_min
    # or ~Vthresh_max. A genuine equilibrium sits clearly inside, both ends.
    assert min(tail) > VTHRESH_MIN + 1.0
    assert max(tail) < VTHRESH_MAX - 1.0

    # And the cell actually fired throughout (an equilibrium, not a dead cell).
    assert runner.window_rate(RUN_MS * 0.75, RUN_MS) > 0.0


# ---------------------------------------------------------------------------
# [AC3] Long enough horizon that Vthresh moved tangibly (no disguised pass)
# ---------------------------------------------------------------------------
def test_ac3_horizon_long_and_vthresh_moved(converged: CellRunner) -> None:
    runner = converged

    # Tens of seconds of simulated time (not hundreds of ms).
    assert runner.times[-1] >= 40000.0

    n = len(runner.Vthresh)
    early_mean = statistics.mean(runner.Vthresh[: n // 4])
    tail_mean = statistics.mean(runner.Vthresh[3 * n // 4:])

    # Homeostasis actually drove Vthresh a tangible distance (started at -53,
    # converged near -48.4). A short run where Vthresh barely moved would fail.
    assert abs(early_mean - tail_mean) >= 2.0


# ---------------------------------------------------------------------------
# [AC4] Tail assertions after the transient settles
# ---------------------------------------------------------------------------
def test_ac4_tail_rate_at_target(converged: CellRunner) -> None:
    tail_rate = converged.window_rate(RUN_MS * 0.75, RUN_MS)
    assert abs(tail_rate - TARGET_HZ) <= 1.0


def test_ac4_tail_vthresh_interior_with_margin(converged: CellRunner) -> None:
    tail_mean = statistics.mean(converged.tail_vthresh(0.25))
    # Clear margin from BOTH bounds (proves equilibrium, not pinning).
    assert VTHRESH_MIN + 1.0 < tail_mean < VTHRESH_MAX - 1.0


def test_ac4_tail_vthresh_settled(converged: CellRunner) -> None:
    tail = converged.tail_vthresh(0.25)
    n = len(converged.Vthresh)
    movement = abs(
        statistics.mean(converged.Vthresh[: n // 4])
        - statistics.mean(tail)
    )
    tail_std = statistics.pstdev(tail)

    # Settled: last-quarter jitter is tiny in absolute terms AND small relative
    # to how far Vthresh travelled during the transient (no longer drifting).
    assert tail_std < 0.3
    assert tail_std < 0.2 * movement


# ---------------------------------------------------------------------------
# [AC5] Survival through silence; spontaneous activity is sub-threshold only
# ---------------------------------------------------------------------------
def test_ac5_survives_silence_and_recovers() -> None:
    cell = _build_cell()
    monitor = ActivityMonitor(window_size=WINDOW_MS)

    def drive(t: float) -> float:
        # drive, then SILENCE for 3 s (> silence threshold), then drive again
        if t < 3000.0 or t >= 6000.0:
            return INPUT_W
        return 0.0

    runner = CellRunner(cell, monitor, dt=DT, input_source=drive)
    runner.run_for(9000.0)

    # No NaN / inf anywhere in the membrane or threshold trajectory.
    assert all(math.isfinite(v) for v in runner.Vm)
    assert all(math.isfinite(v) for v in runner.Vthresh)

    # Deep in the silence window: the cell fired before, is fully silent here,
    # yet spontaneous activity is running (Vm keeps fluctuating, not flatlined),
    # and it NEVER crosses threshold on its own (the Vthresh-0.01 clamp).
    deep = [(t, vm) for t, vm in zip(runner.times, runner.Vm) if 4500.0 <= t < 6000.0]
    deep_spikes = sum(1 for ts in runner.spike_times if 4500.0 <= ts <= 6000.0)
    deep_vm = [vm for _, vm in deep]
    deep_vthresh = [vt for t, vt in zip(runner.times, runner.Vthresh)
                    if 4500.0 <= t < 6000.0]
    assert deep_spikes == 0                              # silent: no firing
    assert statistics.pstdev(deep_vm) > 0.5             # spontaneous is active
    assert max(deep_vm) < min(deep_vthresh)             # stayed sub-threshold

    # Fired before the silence, and recovers (fires again) once input returns.
    assert sum(1 for ts in runner.spike_times if ts < 3000.0) > 0
    assert sum(1 for ts in runner.spike_times if ts >= 6000.0) > 0


# ---------------------------------------------------------------------------
# [AC6] Refractory imposes a hard firing ceiling under flooding input
# ---------------------------------------------------------------------------
def test_ac6_refractory_caps_firing_rate() -> None:
    refractory = 2.0
    cell = Cell(
        neuron_id=1, tau=20.0, target_rate_hz=TARGET_HZ,
        Vthresh=-50.0, Vthresh_min=VTHRESH_MIN, Vthresh_max=VTHRESH_MAX,
        tau_homeostasis=10000.0, refractory_period=refractory,
        spontaneous_silence_threshold_ms=SILENCE_THR_MS, rng_seed=0,
    )
    monitor = ActivityMonitor(window_size=WINDOW_MS)
    # Flood: every step's input is far above any reachable threshold.
    runner = CellRunner(cell, monitor, dt=DT, input_source=lambda t: 1000.0)
    runner.run_for(5000.0)

    tail_rate = runner.window_rate(3000.0, 5000.0)
    physical_cap = 1000.0 / refractory  # 500 Hz

    # The stated physical ceiling is never exceeded...
    assert tail_rate <= physical_cap
    # ...and the cell is genuinely saturated (firing far above target), so this
    # is really testing the ceiling, not an idle cell. The realized ceiling is
    # ~333 Hz here: input arriving during the 2 ms refractory is hard-rejected
    # (Option A), costing one extra dt before the first post-refractory spike,
    # so the true period is refractory + dt = 3 ms rather than 2 ms.
    assert tail_rate > 200.0


# ---------------------------------------------------------------------------
# [Section 5] dt-robustness: TOTAL rate converges across dt (NOT spike-by-spike)
# ---------------------------------------------------------------------------
def test_total_rate_is_approximately_dt_invariant() -> None:
    def converged_rate(dt: float) -> float:
        cell = _build_cell()
        monitor = ActivityMonitor(window_size=WINDOW_MS)
        # Hold PHYSICAL input-per-ms constant: inject INPUT_W * dt each step.
        runner = CellRunner(
            cell, monitor, dt=dt, input_source=lambda t, dt=dt: INPUT_W * dt
        )
        runner.run_for(40000.0)
        return runner.window_rate(30000.0, 40000.0)

    rate_dt1 = converged_rate(1.0)
    rate_dt_half = converged_rate(0.5)

    # Both converge near the setpoint, and the TOTAL rates agree closely — even
    # though the individual spike trains cannot (threshold crossings are
    # quantized to each dt grid, so per-spike matching is neither expected nor
    # asserted here).
    assert abs(rate_dt1 - TARGET_HZ) <= 1.0
    assert abs(rate_dt_half - TARGET_HZ) <= 1.0
    assert abs(rate_dt1 - rate_dt_half) < 0.5
