"""General N-cell Phoenix network — the foundation of the multi-cell phase.

``Network`` generalizes ``TwoCellNetwork`` (which is hardcoded to exactly two
cells and one unidirectional synapse and cannot represent fan-in or fan-out)
to an arbitrary directed graph of cells and synapses. ``TwoCellNetwork`` is
left untouched as the validated 2-cell reference; this is a separate, additive
foundation.

Connectivity is stored as adjacency maps (``outgoing`` / ``incoming``) rather
than a flat synapse list, because the core loop asks two questions on every
spike — "which synapses does this firing cell send to?" (fan-out) and "which
synapses feed this receiving cell?" (fan-in) — and both are certain hot paths.
The pending-delivery queue stays a simple linear-scan list for now (a heap is a
pure, safely-deferrable performance optimization).

Scope: this file validates the STRUCTURE/plumbing of fan-in and fan-out.
Homeostasis, per-cell ``ActivityMonitor``, and spontaneous activity are
intentionally NOT wired into ``step()`` here (same as ``TwoCellNetwork``) —
they enter in a later dynamics step.
"""

from __future__ import annotations

import heapq
import math

from phoenix.cell import Cell
from phoenix.spike import Spike
from phoenix.synapse import Synapse


# A cell within this of Vrest is treated as exactly at rest and may be skipped.
# Chosen far below any physically meaningful voltage (the gap to threshold is
# 25 mV), so the induced error is <= 1e-12 mV — 9 orders below the 1e-9 tolerance
# at which lazy and per-tick integration are asserted identical.
_REST_EPS = 1e-12


def assembly_ignition_voltage(fan_in: int, weight: float) -> float:
    """Voltage delivered to a cell when its fan-in bumps arrive (design-time check).

    In the canonical CONVERGENT ring (delays ``hop, 2*hop, ..., F*hop``) every bump
    lands on the SAME tick, because each earlier predecessor fires proportionally
    sooner. The summed voltage is therefore simply ``fan_in * weight`` — INDEPENDENT
    of ``hop`` and ``tau``, since there is no interval over which the membrane could
    leak between bumps. See the convergent-ring convention in ``Network``.

    Compare against ``Vthresh - Vrest`` (25 mV with the default cell) to check that
    an assembly CAN IGNITE AT ALL, before attributing a dead ring to pruning. A
    sub-threshold ring emits only its injected spikes and stops — it was never alive,
    and mistaking that for a pruning failure is the likeliest misdiagnosis when
    scaling the assembly.

    Pure and stateless. Deliberately NOT wired into ``Network.step()``: this is a
    design-time diagnostic, not part of the simulation loop.
    """
    return fan_in * weight


class Network:
    """An arbitrary directed graph of ``Cell`` nodes connected by ``Synapse`` edges.

    >>> REVERBERATION REQUIRES AN ASSEMBLY (>= ~6 CELLS), NOT A 2-CELL LOOP <<<

    A 2-cell loop is structurally fragile: a SINGLE stray spike kills it
    permanently. Verified autopsy — the intruder makes the post cell fire EARLY;
    that cell enters refractory having already spent its spike; the real loop
    bump then arrives during refractory and is hard-rejected (Option A); the
    chain is broken and NOTHING re-ignites it.

        topology            1 noise hit   10 hits   10% noise
        2-cell loop            0 Hz        0 Hz      45 Hz (broken)
        10-cell assembly     100 Hz      100 Hz     208 Hz

    A 10-cell assembly survives all of it unchanged. Robustness here is an
    EMERGENT STATISTICAL PROPERTY OF REDUNDANCY — the assembly distributes the
    activity, so corrupting one cell does not lose the pattern. Do not go looking
    for a rescue mechanism to bolt onto the 2-cell case:

    - Spontaneous activity was tested and CANNOT rescue it. Its safety clamp
      (``Vm <- min(Vm, Vthresh - 0.01)``) makes it mathematically incapable of
      ever firing a spike, by construction.
    - Partial refractory (Option C) was tested: zero effect.

    >>> ASSEMBLY DESIGN IS DOUBLY CONSTRAINED <<<

    1. THE LAW: ``post_rate * verify_window << 1`` (see ``Synapse.causal_success``).
       The cycle period must keep the firing rate sparse — so ``hop`` must be
       LARGE ENOUGH. Topology sets the rate; homeostasis cannot (reverberation is
       bistable: raising Vthresh extinguishes the loop rather than slowing it).

    2. THE HORIZON: ``FAN_IN * hop <= verify_window``. The DEEPEST synapse in the
       fan-in has latency ``fan_in * hop``. If that exceeds the causal horizon it
       scores causal_success = 0, is decayed away as noise, the fan-in collapses
       (3 -> 2), the summed input drops below the 25 mV firing gap, and the
       assembly DIES. So ``hop`` must also be SMALL ENOUGH.

    THESE PULL IN OPPOSITE DIRECTIONS. With fan_in = 3 and a 10 ms horizon, ``hop``
    is boxed into roughly 2-3 ms. Measured (10-cell ring, fan_in = 3, horizon 10 ms):

        hop   deepest   cs by depth (k1/k2/k3)   outcome
         2      6 ms     1.00 / 1.00 / 1.00      healthy,  50 Hz
         3      9 ms     1.00 / 1.00 / 1.00      healthy,  33 Hz
         4     12 ms     0.99 / 0.99 / 0.00      deepest pruned -> DIES
         5     15 ms     0.99 / 0.99 / 0.00      deepest pruned -> DIES

    The bound is INCLUSIVE: a deepest latency of exactly ``verify_window`` survives
    (verified with fan_in = 2, hop = 5, w = 13: deepest = 10 ms exactly -> cs 0.99,
    alive at 20 Hz; hop = 6 -> 12 ms -> cs 0.00, dies). It is inclusive only because
    the horizon boundary belongs to the HIT — see the boundary convention in
    ``Synapse.resolve_timeouts``; before that was fixed the constraint was
    effectively a strict ``<``.

    THIS IS THE TIGHTEST CONSTRAINT IN THE ARCHITECTURE, and the first thing to
    re-derive when scaling the assembly. Raising ``fan_in`` or ``hop`` requires
    raising ``tau_stdp`` (which widens the horizon), and that in turn weakens noise
    immunity — see the k-sweep in Synapse's verify_window notes.

    Note constraint 2 is fatal exactly when the SURVIVORS are sub-threshold: losing
    the deepest synapse collapses the fan-in F -> F-1, which kills the ring only if
    ``(F - 1) * w < gap``. Verified: (F=2, w=20, hop=8) has a deepest latency of
    16 ms (outside the horizon), so that synapse is pruned to cs = 0.0 and the lone
    survivor delivers 20 mV < 25 mV — the ring DIES.

    >>> CONVERGENT-RING CONVENTION (load-bearing, and previously implicit) <<<

    Cell ``i`` is driven by its ``F`` predecessors with delays
    ``hop, 2*hop, ..., F*hop``. Because predecessor ``i-k`` fires ``(k-1)*hop``
    EARLIER, every bump arrives on the SAME tick:

        source   fires at        delay      arrives at
         i-1     T - 0*hop       1*hop      T + hop
         i-2     T - 1*hop       2*hop      T + hop
         i-3     T - 2*hop       3*hop      T + hop     <- ALL ON ONE TICK

    The longer distance is exactly compensated by the earlier start. This is a
    DELAY LINE — the same mechanism that lets the network discriminate temporal
    order — and it is why fan-in can clear the 25 mV gap at all.

    CONSEQUENCE — in the steady state, ignition is simply:

        F * w  >=  Vthresh - Vrest   (= 25 mV by default)

    INDEPENDENT of ``hop`` and ``tau``: there is no inter-bump interval, so there is
    no inter-bump membrane leak. Use ``assembly_ignition_voltage`` below.

    >>> DO NOT apply a staggered-arrival formula to the STEADY STATE <<<
    ``V = sum(w * exp(-(F - k) * hop / tau))`` mispredicts this topology's steady
    state — it wrongly calls (2,13,3), (2,13,5) and (3,9,2) dead; all three
    reverberate (33.2 / 20.0 / 50.0 Hz).

    BUT that formula is not meaningless: it is the IGNITION-TRANSIENT condition.
    If the ring is ignited ALL AT ONCE (every seed cell at t=0), the first wave has
    no compensating head start, so it genuinely DOES arrive staggered and leak, and
    the transient obeys the staggered sum. Measured:

        F  w  hop | F*w  V_stag | sequential  all-at-once
        3  11   3 |  33   28.62 |    33.2 Hz     33.2 Hz
        2  13   3 |  26   24.19 |    33.2 Hz      0.0 Hz   <- never lights
        2  13   5 |  26   23.12 |    20.0 Hz      0.0 Hz   <- never lights
        2  15   5 |  30   26.68 |    20.0 Hz     20.0 Hz

    So: ``V_stag >= gap`` governs whether an all-at-once ring LIGHTS; ``F * w >= gap``
    governs whether the travelling wave SUSTAINS. They coincide only when both clear
    the gap.

    >>> IGNITION PROCEDURE MATTERS <<<
    Ignite SEQUENTIALLY — ``inject(cell i) at t = i * hop`` — to build the travelling
    wave directly and bypass the staggered transient entirely. Igniting all cells at
    t=0 can leave a perfectly viable ring dark forever.

    A ring that fails to reverberate is NOT necessarily a pruning failure. Check
    ignition FIRST (``F * w >= gap``, and the ignition procedure); a sub-threshold
    ring emits only its injected spikes and stops — it was never alive.
    """

    def __init__(self, dt: float) -> None:
        self.dt: float = dt
        self.current_time: float = 0.0

        self.cells: dict[int, Cell] = {}
        # Adjacency: pre_id -> synapses leaving it; post_id -> synapses arriving.
        self.outgoing: dict[int, list[Synapse]] = {}
        self.incoming: dict[int, list[Synapse]] = {}

        # Pending deliveries: (arrival_time, target_id, effective_weight,
        # source_pre_id), kept as a HEAP keyed on arrival_time. source_pre_id is
        # carried so trace_context is populated correctly on delivery.
        #
        # A linear-scan list looked harmless in single-loop tests (max queue 9),
        # but under distributed activity it explodes — measured max queue 20,499
        # at 400 cells / fan_out 10, with wall time growing ~5.7x for a 4x cell
        # count. Equal arrival_times tie-break on target_id, which is
        # deterministic; never push an un-comparable payload.
        self._pending: list[tuple[float, int, float, int]] = []

        # Cached, sorted neuron ids. Iteration order is deterministic and that is
        # LOAD-BEARING for test reproducibility — but sorting every tick was
        # O(N log N) for nothing. Rebuilt only when cells are added.
        self._sorted_ids: list[int] = []

    def add_cell(self, cell: Cell) -> None:
        """Register a cell by its unique ``neuron_id``."""
        if cell.neuron_id in self.cells:
            raise ValueError(f"duplicate neuron_id {cell.neuron_id}")
        self.cells[cell.neuron_id] = cell
        self.outgoing.setdefault(cell.neuron_id, [])
        self.incoming.setdefault(cell.neuron_id, [])
        self._sorted_ids = sorted(self.cells)  # rebuilt on mutation only

    def add_synapse(self, synapse: Synapse) -> None:
        """Wire a synapse into the adjacency maps (fail loud on dangling ends)."""
        if synapse.pre_id not in self.cells:
            raise ValueError(f"synapse pre_id {synapse.pre_id} is not a registered cell")
        if synapse.post_id not in self.cells:
            raise ValueError(f"synapse post_id {synapse.post_id} is not a registered cell")
        self.outgoing[synapse.pre_id].append(synapse)
        self.incoming[synapse.post_id].append(synapse)

    def inject(self, neuron_id: int, weight: float, source_id: int | None = None) -> None:
        """Drive external stimulus into a cell at the current time."""
        self.cells[neuron_id].receive_input(weight, source_id=source_id)

    def step(self) -> list[Spike]:
        """Advance the whole graph by one tick of duration ``self.dt``.

        Delivery simplification (same as TwoCellNetwork): a pending arrival is
        applied in a single batch at the start of the tick whose interval
        ``[current_time, current_time + dt)`` contains its ``arrival_time`` —
        not at the exact sub-step moment. A newly-enqueued spike (arrival >=
        this tick's end) therefore always waits at least one tick, even at
        zero delay.
        """
        tick_end = self.current_time + self.dt

        # (a) Deliver pending arrivals landing within this tick, passing the
        #     source pre_id so the target's trace_context reflects real origin.
        #     O(log Q) per delivery from the heap, not an O(Q) rescan.
        pending = self._pending
        while pending and pending[0][0] <= tick_end:
            _arrival, target_id, effective_weight, source_pre_id = heapq.heappop(pending)
            self.cells[target_id].receive_input(
                effective_weight, source_id=source_pre_id
            )

        # (b) Integrate cells in a deterministic (neuron_id) order — but ONLY the
        #     ACTIVE ones. Skipping a silent cell is provably exact, not an
        #     approximation:
        #       * A cell at rest with no input CANNOT fire. Leak always moves Vm
        #         TOWARD Vrest (-75), i.e. AWAY from Vthresh (-50), so it can never
        #         cross upward on its own.
        #       * With Vm == Vrest the leak is the identity, so integrate(dt)
        #         reduces to exactly one thing: advancing the clock.
        #     So for a skipped cell we advance its clock and nothing else. The leak
        #     is the closed-form dt-invariant solution, so a cell that sleeps for
        #     1000 ticks and then receives input evolves identically to one that
        #     was integrated every tick.
        #
        #     SCOPE LIMIT: this is safe HERE only because homeostasis and
        #     spontaneous activity are NOT wired into Network.step(). Neither is
        #     dt-invariant (apply_homeostasis scales with dt; maybe_spontaneous_
        #     activity injects noise PER CALL, so the call count matters). CellRunner
        #     must therefore keep integrating every tick — do not port this to it.
        spikes: list[Spike] = []
        for neuron_id in self._sorted_ids:
            cell = self.cells[neuron_id]
            if (
                cell.input_current != 0.0
                or abs(cell.Vm - cell.Vrest) > _REST_EPS
                or cell.t < cell.refractory_until
            ):
                spike = cell.integrate(self.dt)
                if spike is not None:
                    spikes.append(spike)
            else:
                # Provably a no-op apart from the clock. Snap away any residual
                # (<= _REST_EPS) so the cell is exactly at rest from here on.
                cell.Vm = cell.Vrest
                cell.t += self.dt

        # (c) Fan-out: one spike drives every outgoing synapse independently.
        for spike in spikes:
            for synapse in self.outgoing[spike.neuron_id]:
                arrival_time, effective_weight = synapse.propagate(spike)
                heapq.heappush(
                    pending,
                    (arrival_time, synapse.post_id, effective_weight, synapse.pre_id),
                )

        # (d) Advance the clock.
        self.current_time += self.dt

        # NOTE: there is deliberately NO per-tick resolve_timeouts sweep here.
        # Misses are settled at EVENT BOUNDARIES instead — on_pre_spike,
        # on_post_spike and apply_decay each call resolve_timeouts first. That
        # restores the O(spikes) cost model (the sweep was 1.2M calls per 1000
        # ticks on 1200 synapses). See Synapse.resolve_timeouts: DELETING the
        # call, rather than relocating it, would leak _pending_pre without bound
        # and report misses = 0 for a synapse that has failed every time.

        # (d3) LAZY, event-driven weight decay. A synapse's weight is refreshed
        # exactly when it could matter: when its POST cell is active. Idle
        # synapses cost nothing while idle, but are decayed — and therefore
        # pruned — the moment their post cell fires.
        for spike in sorted(spikes, key=lambda s: s.neuron_id):
            for synapse in self.incoming[spike.neuron_id]:
                synapse.apply_decay(self.current_time)

        # (e) Trace-based STDP, generalizing TwoCellNetwork's single-synapse
        #     wiring to arbitrary fan-in/fan-out. For a spiking cell X:
        #       - each INCOMING synapse (partner -> X): X is POST -> on_post_spike
        #       - each OUTGOING synapse (X -> partner): X is PRE  -> on_pre_spike
        #
        # Partner "has it ever spiked?" guard: last_spike_time is initialized to
        # -inf and is NEVER None, so it must be tested with math.isinf, not
        # `is not None` (which would feed -inf into the prediction pathway and
        # produce observed_delay = t - (-inf) = +inf, silently corrupting it).
        #
        # OBSERVATION SEMANTICS — SETTLED (see Synapse.record_observation).
        # The synapse predicts the delay of the TRIGGERING spike: the
        # presynaptic spike most immediately preceding the postsynaptic one,
        # which is exactly what last_spike_time supplies here. All-pairs
        # semantics was evaluated and REJECTED: it cannot converge.
        for spike in spikes:
            x = spike.neuron_id
            for synapse in self.incoming[x]:
                partner = self.cells[synapse.pre_id]
                partner_t = (
                    None
                    if math.isinf(partner.last_spike_time)
                    else partner.last_spike_time
                )
                synapse.on_post_spike(t_post=spike.timestamp, t_pre_partner=partner_t)
            for synapse in self.outgoing[x]:
                partner = self.cells[synapse.post_id]
                partner_t = (
                    None
                    if math.isinf(partner.last_spike_time)
                    else partner.last_spike_time
                )
                synapse.on_pre_spike(t_pre=spike.timestamp, t_post_partner=partner_t)

        # (f) Return this tick's spikes in chronological order.
        return sorted(spikes, key=lambda s: s.timestamp)

    def run(self, n_steps: int) -> list[Spike]:
        """Run ``n_steps`` ticks, returning all spikes chronologically."""
        all_spikes: list[Spike] = []
        for _ in range(n_steps):
            all_spikes.extend(self.step())
        return all_spikes

    def weight(self, pre_id: int, post_id: int) -> float:
        """Current weight of the ``pre_id -> post_id`` synapse (diagnostic).

        Linear search over ``outgoing[pre_id]`` — fine at small scale; this is
        an inspection accessor, not part of the hot loop.
        """
        for synapse in self.outgoing[pre_id]:
            if synapse.post_id == post_id:
                return synapse.weight
        raise ValueError(f"no synapse {pre_id} -> {post_id}")
