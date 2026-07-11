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

import math

from phoenix.cell import Cell
from phoenix.spike import Spike
from phoenix.synapse import Synapse


class Network:
    """An arbitrary directed graph of ``Cell`` nodes connected by ``Synapse`` edges."""

    def __init__(self, dt: float) -> None:
        self.dt: float = dt
        self.current_time: float = 0.0

        self.cells: dict[int, Cell] = {}
        # Adjacency: pre_id -> synapses leaving it; post_id -> synapses arriving.
        self.outgoing: dict[int, list[Synapse]] = {}
        self.incoming: dict[int, list[Synapse]] = {}

        # Pending deliveries: (arrival_time, target_id, effective_weight,
        # source_pre_id). source_pre_id is carried so trace_context is populated
        # correctly on delivery, exactly as TwoCellNetwork does.
        self._pending: list[tuple[float, int, float, int]] = []

    def add_cell(self, cell: Cell) -> None:
        """Register a cell by its unique ``neuron_id``."""
        if cell.neuron_id in self.cells:
            raise ValueError(f"duplicate neuron_id {cell.neuron_id}")
        self.cells[cell.neuron_id] = cell
        self.outgoing.setdefault(cell.neuron_id, [])
        self.incoming.setdefault(cell.neuron_id, [])

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
        still_pending: list[tuple[float, int, float, int]] = []
        for arrival_time, target_id, effective_weight, source_pre_id in self._pending:
            if arrival_time <= tick_end:
                self.cells[target_id].receive_input(
                    effective_weight, source_id=source_pre_id
                )
            else:
                still_pending.append(
                    (arrival_time, target_id, effective_weight, source_pre_id)
                )
        self._pending = still_pending

        # (b) Integrate every cell in a deterministic (neuron_id) order.
        spikes: list[Spike] = []
        for neuron_id in sorted(self.cells):
            spike = self.cells[neuron_id].integrate(self.dt)
            if spike is not None:
                spikes.append(spike)

        # (c) Fan-out: one spike drives every outgoing synapse independently.
        for spike in spikes:
            for synapse in self.outgoing[spike.neuron_id]:
                arrival_time, effective_weight = synapse.propagate(spike)
                # TODO: replace pending-queue linear scan with a heap when N grows
                self._pending.append(
                    (arrival_time, synapse.post_id, effective_weight, synapse.pre_id)
                )

        # (d) Advance the clock.
        self.current_time += self.dt

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
        # NOTE: last_spike_time pairing preserves the known [N1] single-pairing
        # limitation (see review). Weight STDP already uses full traces; only
        # the prediction/observation pathway pairs single-last. Revisit when
        # observation semantics are settled. Not final.
        #
        # (Recurrent-topology property, not handled now: if a synapse X->Y has
        # BOTH endpoints fire in the same tick, that synapse receives BOTH
        # on_pre_spike and on_post_spike this step. This matches TwoCellNetwork
        # and cannot occur in fan-in 2->1, so it is left as-is.)
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
