"""Minimal two-cell, one-directional Phoenix simulation loop.

Hybrid design: a fixed timestep drives each cell's continuous leak/input
dynamics via ``integrate(dt)``, while spike delivery across a ``Synapse``
(which carries its own propagation delay) is handled through a small
pending-delivery queue rather than applied instantly.

This is deliberately minimal — a single ``Synapse`` from ``cell_a`` to
``cell_b`` — not a general N-cell graph. That generalization (and a heap
instead of a linear-scan queue, if the number of pending deliveries grows)
is a later step.
"""

from __future__ import annotations

import math

from phoenix.cell import Cell
from phoenix.spike import Spike
from phoenix.synapse import Synapse


class TwoCellNetwork:
    """Drives ``cell_a`` and ``cell_b`` in lockstep, connected by one ``Synapse``.

    ``synapse.pre_id`` is assumed to be ``cell_a.neuron_id`` and
    ``synapse.post_id`` is assumed to be ``cell_b.neuron_id`` — propagation
    only flows a→b; ``cell_b``'s spikes are not propagated anywhere.
    """

    def __init__(self, cell_a: Cell, cell_b: Cell, synapse: Synapse, dt: float) -> None:
        self.cell_a: Cell = cell_a
        self.cell_b: Cell = cell_b
        self.synapse: Synapse = synapse
        self.dt: float = dt
        self.current_time: float = 0.0

        # Pending spike deliveries not yet applied to their target cell.
        # (arrival_time, target_cell_id, effective_weight) tuples, linear-
        # scanned each tick — fine at this scale, would want a heap for N cells.
        self._pending: list[tuple[float, int, float]] = []

    def _cell_for_id(self, cell_id: int) -> Cell:
        if cell_id == self.cell_a.neuron_id:
            return self.cell_a
        if cell_id == self.cell_b.neuron_id:
            return self.cell_b
        raise ValueError(f"No cell with neuron_id={cell_id} in this network")

    @property
    def current_weight(self) -> float:
        """Read-only convenience view of the connecting synapse's weight."""
        return self.synapse.weight

    def step(self) -> list[Spike]:
        """Advance the simulation by one tick of duration ``self.dt``.

        Known simplification: deliveries are applied in a single batch at
        the *start* of the tick whose interval ``[current_time,
        current_time + dt)`` contains their ``arrival_time`` — not at the
        exact sub-step moment they'd physically arrive. This is a
        tick-granularity approximation, not a bug; a delay smaller than
        ``dt`` is still respected in that it cannot be delivered before its
        tick arrives, only coarsened to that tick's boundary.

        STDP: each new spike on either cell drives one O(1) trace-based
        update via ``Synapse.on_pre_spike``/``on_post_spike`` (Song, Miller
        & Abbott 2000 all-to-all trace scheme) rather than scanning the
        other cell's full spike history — the traces already aggregate
        every past same-type spike's contribution, decaying continuously,
        so a single call captures what a history scan used to require a
        loop for. ``Cell.spike_history`` itself is untouched (still useful
        for diagnostics) but is no longer read by this method.
        """
        spikes: list[Spike] = []
        tick_end = self.current_time + self.dt

        # (a) Deliver anything whose arrival falls within this tick. Source
        # identity (self.synapse.pre_id) is passed through so the target
        # cell's trace_context reflects which presynaptic cell the input
        # actually came from, not just that something arrived.
        still_pending: list[tuple[float, int, float]] = []
        for arrival_time, target_id, effective_weight in self._pending:
            if arrival_time <= tick_end:
                self._cell_for_id(target_id).receive_input(
                    effective_weight, source_id=self.synapse.pre_id
                )
            else:
                still_pending.append((arrival_time, target_id, effective_weight))
        self._pending = still_pending

        # (b) Integrate cell_a; propagate any resulting spike via the synapse.
        spike_a = self.cell_a.integrate(self.dt)
        if spike_a is not None:
            spikes.append(spike_a)
            arrival_time, effective_weight = self.synapse.propagate(spike_a)
            self._pending.append((arrival_time, self.synapse.post_id, effective_weight))

        # (c) Integrate cell_b independently. One-directional network: any
        # spike cell_b produces is returned but never propagated further.
        spike_b = self.cell_b.integrate(self.dt)
        if spike_b is not None:
            spikes.append(spike_b)

        # (d) Advance the clock.
        self.current_time += self.dt

        # (e) STDP: one O(1) trace update per new spike this tick. The
        # "partner" time passed in is simply the other cell's current
        # last_spike_time (None if it has never spiked) — used only to
        # compute prediction modulation, since the trace itself already
        # carries the aggregated magnitude across all past same-type spikes.
        if spike_a is not None:
            t_post_partner = (
                None
                if math.isinf(self.cell_b.last_spike_time)
                else self.cell_b.last_spike_time
            )
            self.synapse.on_pre_spike(t_pre=spike_a.timestamp, t_post_partner=t_post_partner)
        if spike_b is not None:
            t_pre_partner = (
                None
                if math.isinf(self.cell_a.last_spike_time)
                else self.cell_a.last_spike_time
            )
            self.synapse.on_post_spike(t_post=spike_b.timestamp, t_pre_partner=t_pre_partner)

        return spikes

    def run(self, n_steps: int) -> list[Spike]:
        """Run ``n_steps`` ticks, returning all spikes in chronological order."""
        all_spikes: list[Spike] = []
        for _ in range(n_steps):
            all_spikes.extend(self.step())
        return all_spikes
