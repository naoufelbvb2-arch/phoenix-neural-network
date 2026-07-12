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
    """

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

        # (d2) Score any pre-spike whose verification window has expired as a
        # MISS. This is the ONLY place misses are counted, so it must run every
        # tick for every synapse — otherwise a synapse only ever sees its
        # successes and causal_success degenerates into confirmation bias.
        # Deterministic order (by pre_id, then insertion order).
        for neuron_id in sorted(self.outgoing):
            for synapse in self.outgoing[neuron_id]:
                synapse.resolve_timeouts(self.current_time)

        # (d3) LAZY, event-driven weight decay. A synapse's weight is refreshed
        # exactly when it could matter: when its POST cell is active. Idle
        # synapses cost nothing while idle, but are decayed — and therefore
        # pruned — the moment their post cell fires. Decay is computed from
        # elapsed time in one step, so skipping ticks is exact, not approximate.
        #
        # Deliberately NOT a sweep over all synapses every tick: that would be
        # O(synapses x ticks) and would destroy the event-driven cost model.
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
        # which is exactly what last_spike_time supplies here. This is a
        # deliberate, tested modeling choice, not an implementation artifact.
        # All-pairs semantics was evaluated and REJECTED: it cannot converge —
        # averaging structurally distinct delays yields an expectation matching
        # no real physical delay, pinning prediction_error permanently above
        # zero (see test_prediction_error_converges_to_zero_on_perfect_pattern
        # and test_all_pairs_semantics_cannot_converge). Weights may still SUM
        # over all pairs via traces; prediction must ESTIMATE A DISTRIBUTION,
        # which requires a homogeneous sample.
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
