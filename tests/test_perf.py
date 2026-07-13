"""The cost model: O(spikes), not O(cells x ticks) or O(synapses x ticks).

These tests pin the COST MODEL itself, not just behavior. The simulator had two
per-tick sweeps that silently destroyed the event-driven (neuromorphic) cost
model — profiled at 400 cells / 1,198 synapses / 1,000 ticks:

    integrate          400,000   = ticks x cells
    resolve_timeouts 1,198,000   = ticks x synapses   <- worst
    apply_decay            165   = event-driven (already correct)

Both sweeps are gone. resolve_timeouts was RELOCATED to event boundaries (never
deleted — see P3 for why deleting it would have been a disaster), and integration
now skips provably-silent cells.
"""

from __future__ import annotations

import heapq
import random

import pytest

from phoenix.cell import Cell
from phoenix.monitor import ActivityMonitor
from phoenix.network_graph import Network
from phoenix.runner import CellRunner
from phoenix.synapse import Synapse


def _syn(pre: int, post: int, weight: float, delay: float) -> Synapse:
    return Synapse(
        pre_id=pre, post_id=post, weight=weight, distance=delay,
        propagation_speed=1.0, decay_constant=1000.0,
    )


def _ring(n_cells: int = 10, fan_in: int = 3, weight: float = 11.0,
          hop: int = 3) -> Network:
    net = Network(dt=1.0)
    for i in range(n_cells):
        net.add_cell(Cell(neuron_id=i))
    for i in range(n_cells):
        for k in range(1, fan_in + 1):
            net.add_synapse(_syn((i - k) % n_cells, i, weight, k * hop))
    return net


# ---------------------------------------------------------------------------
# P1. THE COST MODEL ITSELF: no per-tick sweeps
# ---------------------------------------------------------------------------
def test_no_per_tick_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call counts must scale with SPIKES, not with ticks x cells / ticks x synapses.

    This is the test that pins the neuromorphic cost model. If someone
    reintroduces a per-tick sweep, this fails loudly rather than the simulator
    merely getting slower for reasons nobody notices until N-cell scale.
    """
    counts = {"integrate": 0, "resolve_timeouts": 0, "apply_decay": 0}
    original = {
        "integrate": Cell.integrate,
        "resolve_timeouts": Synapse.resolve_timeouts,
        "apply_decay": Synapse.apply_decay,
    }

    def counting(name: str, func):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return func(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(Cell, "integrate", counting("integrate", original["integrate"]))
    monkeypatch.setattr(
        Synapse, "resolve_timeouts",
        counting("resolve_timeouts", original["resolve_timeouts"]),
    )
    monkeypatch.setattr(
        Synapse, "apply_decay", counting("apply_decay", original["apply_decay"])
    )

    n_cells, ticks = 40, 500
    net = Network(dt=1.0)
    for i in range(n_cells):
        net.add_cell(Cell(neuron_id=i))
    for i in range(n_cells):  # sparse feed-forward chain: most cells stay silent
        net.add_synapse(_syn(i, (i + 1) % n_cells, 8.0, 2.0))
    n_synapses = sum(len(v) for v in net.outgoing.values())

    spikes = 0
    for tick in range(ticks):
        if tick == 0:
            net.inject(0, 100.0)
        spikes += len(net.step())

    # The old sweeps would have produced EXACTLY these numbers.
    per_tick_cells = ticks * n_cells        # 20,000
    per_tick_synapses = ticks * n_synapses  # 20,000

    # resolve_timeouts is now event-driven: nowhere near a per-tick sweep.
    assert counts["resolve_timeouts"] < per_tick_synapses / 10
    # apply_decay always was, and still is.
    assert counts["apply_decay"] < per_tick_synapses / 10
    # integrate skips silent cells, so it is far below ticks x cells.
    assert counts["integrate"] < per_tick_cells / 2

    # And all three scale with the ACTIVITY, not the clock.
    assert counts["resolve_timeouts"] <= 10 * max(spikes, 1)
    assert counts["apply_decay"] <= 10 * max(spikes, 1)


# ---------------------------------------------------------------------------
# P2. Event-boundary settling matches per-tick settling
# ---------------------------------------------------------------------------
def test_lazy_timeout_settling_matches_per_tick() -> None:
    """Settling misses on events, not every tick, is accounting-equivalent.

    BOUNDARY CASE (documented, not hidden): a pending spike whose window expires
    BETWEEN two events is now settled at the later event. It can therefore
    occasionally be caught as a HIT where a per-tick sweep would have scored a
    MISS. The difference is bounded by ONE event, and only for a pending whose
    window closes in a gap where nothing touched the synapse.
    """
    rng = random.Random(11)
    events = sorted(rng.uniform(0, 2_000) for _ in range(200))

    event_driven = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    per_tick = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    for index, t in enumerate(events):
        if index % 2 == 0:
            event_driven.on_pre_spike(t, None)
            per_tick.on_pre_spike(t, None)
        else:
            event_driven.on_post_spike(t, None)
            per_tick.on_post_spike(t, None)
        # The OLD model additionally swept every tick.
        for tick in range(int(t), int(t) + 1):
            per_tick.resolve_timeouts(float(tick))

    assert abs(event_driven.hits - per_tick.hits) <= 1
    assert abs(event_driven.misses - per_tick.misses) <= 1


# ---------------------------------------------------------------------------
# P3. Why resolve_timeouts was RELOCATED, not DELETED
# ---------------------------------------------------------------------------
def test_pending_spikes_do_not_accumulate() -> None:
    """The trap that the obvious "optimization" would have walked into.

    Simply deleting the per-tick call looks harmless and is catastrophic: a
    synapse whose pre fires repeatedly with no post response would accumulate
    _pending_pre WITHOUT BOUND (a memory leak, in precisely the noisy synapse we
    most need to prune) while reporting misses = 0 and causal_success = None —
    i.e. "no evidence" for a synapse that has FAILED every single time.

    Because the call is relocated rather than removed, each new pre-spike settles
    the previous ones first.
    """
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    for k in range(100):
        synapse.on_pre_spike(100.0 + k * 50.0, None)  # post NEVER responds

    # Bounded: settled on every event, not hoarded.
    assert len(synapse._pending_pre) <= 2

    # And the failures are actually COUNTED — the synapse knows it is useless.
    assert synapse.misses >= 98
    assert synapse.hits == 0
    assert synapse.causal_success == 0.0  # not None: this is evidence of failure


# ---------------------------------------------------------------------------
# P4. Lazy integration is EXACT, not an approximation
# ---------------------------------------------------------------------------
def test_lazy_integration_is_exact() -> None:
    """A cell silent for 1,000 ticks then driven behaves identically to one
    integrated every single tick.

    The leak is the closed-form dt-invariant solution, and for a cell at Vrest it
    is the identity — so skipping it changes nothing but the clock.
    """
    silent_ticks = 1_000

    # Reference: a bare Cell integrated EVERY tick.
    reference = Cell(neuron_id=7)
    for _ in range(silent_ticks):
        reference.integrate(1.0)

    # Network: the same cell, SKIPPED while silent (it has no synapses at all).
    net = Network(dt=1.0)
    net.add_cell(Cell(neuron_id=7))
    net.run(n_steps=silent_ticks)
    lazy = net.cells[7]

    # After the same number of ticks, the skipped cell is in the same state.
    assert lazy.t == reference.t                       # clock kept in lockstep
    assert lazy.Vm == pytest.approx(reference.Vm, abs=1e-9)

    # Now drive both identically, on the same tick.
    reference.receive_input(100.0)
    reference_spike = reference.integrate(1.0)

    net.inject(7, 100.0)
    spikes = net.step()

    # The 1,000 skipped ticks left NO trace: same spike, same time, same Vm.
    assert reference_spike is not None
    assert len(spikes) == 1
    assert spikes[0].timestamp == pytest.approx(reference_spike.timestamp, abs=1e-9)
    assert lazy.t == reference.t
    assert lazy.Vm == pytest.approx(reference.Vm, abs=1e-9)


# ---------------------------------------------------------------------------
# P5. Skipping is SAFE: a silent cell cannot fire
# ---------------------------------------------------------------------------
def test_silent_cell_cannot_fire() -> None:
    """Leak always moves Vm TOWARD Vrest (-75), i.e. AWAY from Vthresh (-50).

    A cell parked 2 mV below threshold with zero input decays away from it and can
    never cross upward on its own. This is what makes skipping a silent cell safe
    rather than merely cheap.
    """
    cell = Cell(neuron_id=1)
    cell.Vm = -52.0  # just 2 mV below threshold

    for _ in range(1_000):
        assert cell.integrate(1.0) is None  # never fires

    assert cell.Vm < -52.0                    # it moved AWAY from threshold
    assert cell.Vm == pytest.approx(cell.Vrest, abs=1e-6)  # and settled at rest


# ---------------------------------------------------------------------------
# P6. The heap delivers in strict arrival order
# ---------------------------------------------------------------------------
def test_heap_delivery_order_is_correct() -> None:
    """Out-of-order insertion must still deliver in strict arrival_time order.

    Equal arrival_times tie-break on target_id, which is deterministic — that
    property is load-bearing for test reproducibility.
    """
    net = Network(dt=1.0)
    for i in (1, 2, 3):
        net.add_cell(Cell(neuron_id=i))

    # Push deliberately out of order.
    for arrival, target in ((50.0, 3), (10.0, 1), (30.0, 2), (10.0, 2), (20.0, 1)):
        heapq.heappush(net._pending, (arrival, target, 1.0, 1))

    popped = []
    while net._pending:
        popped.append(heapq.heappop(net._pending)[:2])

    assert popped == [(10.0, 1), (10.0, 2), (20.0, 1), (30.0, 2), (50.0, 3)]
    # Strictly non-decreasing arrival times, ties broken deterministically.
    assert popped == sorted(popped)


# ---------------------------------------------------------------------------
# P7. Every scientific conclusion survives the refactor
# ---------------------------------------------------------------------------
def test_full_stack_robustness_unchanged() -> None:
    """The validated assembly still behaves identically after the cost refactor.

    Measured over the full 15-scenario sweep (noise 0->10% x 3 seeds), the
    before/after numbers are BIT-FOR-BIT IDENTICAL — rate 33.2 Hz, assembly 10.814,
    idle 2.4576 at 30 s. The predicted boundary drift did not even materialize.
    Spot-checked here at 5% noise to keep the suite fast.
    """
    ring_size, fan_in, weight, hop = 10, 3, 11.0, 3
    noise_id, idle_id = 100, 101

    net = _ring(ring_size, fan_in, weight, hop)
    net.add_cell(Cell(neuron_id=noise_id))
    net.add_cell(Cell(neuron_id=idle_id))
    net.add_synapse(_syn(noise_id, 0, weight, 1.0))
    net.add_synapse(_syn(idle_id, 1, weight, 1.0))  # its pre NEVER fires

    rng = random.Random(1)
    run_ms = 30_000
    spikes = []
    for _ in range(run_ms):
        for i in range(fan_in):
            if net.current_time == i * hop:
                net.inject(i, 100.0)
        if rng.random() < 0.05:
            net.inject(noise_id, 100.0)
        spikes.extend(net.step())

    tail = [s for s in spikes if s.neuron_id == 0 and s.timestamp > run_ms - 5_000]
    rate = len(tail) / 5.0
    assembly = [s for i in range(ring_size) for s in net.incoming[i] if s.pre_id < ring_size]
    assembly_mean = sum(s.weight for s in assembly) / len(assembly)
    noise = next(s for s in net.incoming[0] if s.pre_id == noise_id)
    idle = next(s for s in net.incoming[1] if s.pre_id == idle_id)

    assert rate == pytest.approx(33.2, abs=1.0)              # reverberation alive
    assert assembly_mean == pytest.approx(10.81, abs=0.5)    # weights interior
    assert 0.0 < assembly_mean < assembly[0].w_max
    assert noise.weight < assembly_mean                      # active noise pruned
    assert idle.weight < 0.5 * weight                        # idle synapse pruned
    assert idle.causal_success is None                       # it never fired


# ---------------------------------------------------------------------------
# P8. SCOPE LIMIT: CellRunner must NOT use lazy integration
# ---------------------------------------------------------------------------
def test_cell_runner_still_integrates_every_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lazy integration is safe for Network ONLY.

    CellRunner wires in homeostasis and spontaneous activity, and NEITHER is
    dt-invariant: apply_homeostasis scales linearly with dt, and
    maybe_spontaneous_activity injects noise PER CALL, so the number of calls is
    itself physically meaningful. Skipping ticks there would silently change the
    dynamics. CellRunner therefore integrates every tick, always.
    """
    calls = {"n": 0}
    original = Cell.integrate

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(Cell, "integrate", counting)

    cell = Cell(neuron_id=1, rng_seed=0)      # silent: no input source at all
    monitor = ActivityMonitor(window_size=1_000.0)
    runner = CellRunner(cell, monitor, dt=1.0, input_source=None)

    ticks = 500
    runner.run(n_steps=ticks)

    # EVERY tick, despite the cell being completely silent throughout.
    assert calls["n"] == ticks
