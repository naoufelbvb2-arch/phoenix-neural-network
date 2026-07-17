"""Struct-of-Arrays vectorized compute layer — numerically identical to the OOP core.

This is a PARALLEL execution layer, not a replacement. ``Cell`` / ``Synapse`` /
``Network`` remain the mathematical oracle; ``SoANetwork`` is a vectorized
implementation of the SAME dynamics, validated against them bit-for-bit. Keeping
both preserves the ability to diagnose and to prove equivalence continuously.

SCOPE — PHASE 1: DYNAMICS ONLY (voltage, firing, delayed delivery). Learning
(STDP / CST / decay) is intentionally NOT ported here; it is per-synapse,
history-dependent, and hard to vectorize. It arrives in a separate SoA-2 step,
only after dynamics equivalence is proven. So a matching OOP reference must have
its learning frozen (``learning_rate=0`` and ``tau_decay=inf``), or use bare
``Cell`` objects with no synapses at all.

>>> FOUR CRITICAL ORDERING DETAILS (each is bit-exact-or-nothing) <<<
1. integrate order: leak `Vm = Vrest + (Vm-Vrest)*exp(-dt/tau)`, THEN add the
   accepted input, THEN advance the clock `t += dt`, THEN the threshold check.
2. `t` advances BEFORE the threshold check. A crossing is recorded at the
   post-increment `t`, and `refractory_until = t + refractory_period` uses it too.
3. Option A: input arriving while `t < refractory_until` is DROPPED ENTIRELY
   (not deferred), using the PRE-increment `t`.
4. Refractory is tested with a DIFFERENT `t` in two places: input rejection uses
   the pre-increment `t`; firing suppression uses the post-increment `t`.

float64 everywhere — bit-for-bit matching requires it; float32 would drift.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


def _as_f64_array(value: float | np.ndarray, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar to length ``n``, or validate an array, as float64."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n, float(array), dtype=np.float64)
    if array.shape != (n,):
        raise ValueError(f"{name}: expected scalar or shape ({n},), got {array.shape}")
    return array.astype(np.float64, copy=True)


class CellArrays:
    """Membrane dynamics for ``n`` cells, one array index per cell.

    A single scalar clock ``t`` (all cells advance in lockstep, exactly as the
    OOP ``Network`` keeps every cell at ``current_time``). Per-cell parameters
    may be scalar (uniform) or length-``n`` arrays (heterogeneous, e.g. ``tau``).
    """

    def __init__(
        self,
        n: int,
        *,
        Vrest: float | np.ndarray = -75.0,
        Vthresh: float | np.ndarray = -50.0,
        Vreset: float | np.ndarray = -75.0,
        tau: float | np.ndarray = 20.0,
        refractory_period: float | np.ndarray = 2.0,
        dt: float = 1.0,
    ) -> None:
        self.n = n
        self.dt = float(dt)

        self.Vrest = _as_f64_array(Vrest, n, "Vrest")
        self.Vthresh = _as_f64_array(Vthresh, n, "Vthresh")
        self.Vreset = _as_f64_array(Vreset, n, "Vreset")
        self.tau = _as_f64_array(tau, n, "tau")
        self.refractory_period = _as_f64_array(refractory_period, n, "refractory_period")

        # Precomputed per-cell leak factor exp(-dt/tau).
        #
        # BIT-EXACTNESS: the OOP core computes `math.exp(-dt/tau)` per integrate
        # call. `math.exp` and `np.exp` may differ by 1 ULP, and at a cell sitting
        # exactly on threshold that single ULP can flip a spike and diverge the
        # whole train. dt and tau are constant, so we compute the factor ONCE,
        # element-by-element with the SAME `math.exp` the oracle uses — making the
        # leak provably identical rather than coincidentally close (and removing
        # exp from the per-tick hot path entirely).
        self._leak = np.array(
            [math.exp(-self.dt / t) for t in self.tau], dtype=np.float64
        )

        # Mutable state.
        self.Vm = self.Vrest.copy()
        self.refractory_until = np.zeros(n, dtype=np.float64)
        self.last_spike_time = np.full(n, -np.inf, dtype=np.float64)
        self.t = 0.0  # scalar simulation clock

    def step(self, raw_input: np.ndarray) -> np.ndarray:
        """Advance every cell one tick, in the exact OOP order. Returns fired indices.

        ``raw_input[i]`` is the total input staged for cell ``i`` this tick (the
        sum of the bumps that OOP would have delivered via ``receive_input`` plus
        any external injection). Option-A rejection is all-or-nothing per cell per
        tick — OOP evaluates every bump against the SAME (constant, pre-increment)
        ``t`` and ``refractory_until``, so zeroing the whole cell's input when it
        is refractory is exactly equivalent.
        """
        # (3)+(4a) Option A rejection with the PRE-increment clock.
        accepted = np.where(self.t < self.refractory_until, 0.0, raw_input)

        # (1) exact exponential leak (precomputed factor, == math.exp(-dt/tau)),
        #     THEN accepted input.
        self.Vm = self.Vrest + (self.Vm - self.Vrest) * self._leak
        self.Vm += accepted

        # (2) advance the clock BEFORE the threshold check.
        self.t += self.dt

        # (4b) firing suppression with the POST-increment clock.
        fired = (self.t >= self.refractory_until) & (self.Vm >= self.Vthresh)

        # (2) record + reset, all at the post-increment t. The per-cell parameter
        #     arrays (refractory_period, Vreset) MUST be indexed by `fired` too —
        #     the RHS length must match the number of fired cells, not n.
        self.last_spike_time[fired] = self.t
        self.refractory_until[fired] = self.t + self.refractory_period[fired]
        self.Vm[fired] = self.Vreset[fired]

        return np.nonzero(fired)[0]


class SoANetwork:
    """Vectorized N-cell network: SoA cell dynamics + a delayed-delivery queue.

    Mirrors ``phoenix.network_graph.Network`` for DYNAMICS ONLY. Synapse weights
    are frozen (no STDP/decay in phase 1), so the matching OOP reference must
    freeze learning too. The pending-delivery heap uses the SAME entry shape and
    tie-break as the OOP heap — ``(arrival_time, target_id, effective_weight,
    source_id)`` — so bumps pop in identical order and are accumulated in identical
    order, which is what makes the float64 sums bit-exact (float addition is not
    associative, so summation ORDER is load-bearing).
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = float(dt)
        self._cell_specs: list[tuple[int, dict]] = []
        self._syn_specs: list[tuple[int, int, float, float, float, float]] = []
        self._built = False

        # Populated by _build(). Synapses are stored as PARALLEL ARRAYS in CSR
        # form (grouped by source index): _syn_indptr[idx]:_syn_indptr[idx+1]
        # slices out a cell's outgoing edges. This is the memory-efficient SoA
        # representation — ~24 B/synapse (two int32 + two float64) versus ~792 B
        # for an OOP Synapse — while still giving O(fan-out) delivery.
        self._ids: list[int] = []
        self._id2idx: dict[int, int] = {}
        self.cells: CellArrays | None = None
        self._syn_indptr: np.ndarray | None = None    # (n+1,) int64 CSR offsets
        self._syn_post_id: np.ndarray | None = None   # (E,) int32 target neuron_id
        self._syn_eff_w: np.ndarray | None = None      # (E,) float64 attenuated weight
        self._syn_delay: np.ndarray | None = None      # (E,) float64 delay
        self._heap: list[tuple[float, int, float, int]] = []  # (arrival, target_id, eff_w, source_id)
        self._external: np.ndarray | None = None       # staged injections

    def add_cell(self, neuron_id: int, **params: float) -> None:
        if self._built:
            raise RuntimeError("cannot add cells after the network is built")
        if any(nid == neuron_id for nid, _ in self._cell_specs):
            raise ValueError(f"duplicate neuron_id {neuron_id}")
        self._cell_specs.append((neuron_id, params))

    def add_synapse(
        self, pre_id: int, post_id: int, weight: float, distance: float,
        propagation_speed: float = 1.0, decay_constant: float = 10.0,
    ) -> None:
        if self._built:
            raise RuntimeError("cannot add synapses after the network is built")
        self._syn_specs.append(
            (pre_id, post_id, weight, distance, propagation_speed, decay_constant)
        )

    def _build(self) -> None:
        ids = sorted(nid for nid, _ in self._cell_specs)
        self._ids = ids
        self._id2idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)

        params_by_id = dict(self._cell_specs)

        def gather(key: str, default: float) -> np.ndarray:
            return np.array(
                [params_by_id[nid].get(key, default) for nid in ids], dtype=np.float64
            )

        self.cells = CellArrays(
            n,
            Vrest=gather("Vrest", -75.0),
            Vthresh=gather("Vthresh", -50.0),
            Vreset=gather("Vreset", -75.0),
            tau=gather("tau", 20.0),
            refractory_period=gather("refractory_period", 2.0),
            dt=self.dt,
        )

        # Build parallel synapse arrays, sorted by source index (CSR).
        edges = []  # (pre_idx, post_id, eff_w, delay)
        for pre_id, post_id, weight, distance, prop_speed, decay in self._syn_specs:
            if pre_id not in self._id2idx:
                raise ValueError(f"synapse pre_id {pre_id} is not a registered cell")
            if post_id not in self._id2idx:
                raise ValueError(f"synapse post_id {post_id} is not a registered cell")
            delay = distance / prop_speed
            eff_w = weight * math.exp(-distance / decay)
            edges.append((self._id2idx[pre_id], post_id, eff_w, delay))

        # Stable sort by source index preserves per-source insertion order, so
        # fan-out enqueues edges in the same order the OOP adjacency list holds
        # them — keeping heap tie-breaks (and thus delivery order) identical.
        edges.sort(key=lambda e: e[0])
        e = len(edges)
        self._syn_post_id = np.array([x[1] for x in edges], dtype=np.int32)
        self._syn_eff_w = np.array([x[2] for x in edges], dtype=np.float64)
        self._syn_delay = np.array([x[3] for x in edges], dtype=np.float64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        for pre_idx, _post, _w, _d in edges:
            indptr[pre_idx + 1] += 1
        self._syn_indptr = np.cumsum(indptr)

        self._external = np.zeros(n, dtype=np.float64)
        self._built = True

    def _ensure_built(self) -> None:
        if not self._built:
            self._build()

    def inject(self, neuron_id: int, weight: float) -> None:
        """Stage external input for the NEXT ``step`` (folded into that tick's raw)."""
        self._ensure_built()
        self._external[self._id2idx[neuron_id]] += weight

    def step(self) -> list[int]:
        """One tick. Returns the neuron_ids that fired, ascending (deterministic)."""
        self._ensure_built()
        cells = self.cells
        tick_end = cells.t + self.dt

        # (a) Accumulate arrivals landing this tick, in EXACT heap-pop order, plus
        #     the staged injections. Scalar accumulation in pop order guarantees
        #     the summation order matches OOP's `input_current += weight` (np.add.at
        #     would be the order-independent vectorized form; we deliberately keep
        #     ordered accumulation because bit-exactness needs a fixed order).
        raw = self._external.copy()
        self._external[:] = 0.0
        heap = self._heap
        while heap and heap[0][0] <= tick_end:
            _arrival, target_id, eff_w, _source_id = heapq.heappop(heap)
            raw[self._id2idx[target_id]] += eff_w

        # (b) Vectorized cell dynamics.
        fired_idx = cells.step(raw)

        # (c) Fan-out: enqueue each fired cell's outgoing bumps, mirroring the OOP
        #     heap entry shape/tie-break so future pops match order-for-order.
        new_t = cells.t  # post-increment spike timestamp
        indptr, post_id, eff_w, delay = (
            self._syn_indptr, self._syn_post_id, self._syn_eff_w, self._syn_delay,
        )
        for idx in fired_idx:
            source_id = self._ids[idx]
            for j in range(indptr[idx], indptr[idx + 1]):
                heapq.heappush(
                    heap, (new_t + delay[j], int(post_id[j]), eff_w[j], source_id)
                )

        return [self._ids[i] for i in fired_idx]

    def run(self, n_steps: int) -> list[tuple[int, float]]:
        """Run ``n_steps`` ticks; return the full (neuron_id, timestamp) spike train."""
        self._ensure_built()
        train: list[tuple[int, float]] = []
        for _ in range(n_steps):
            t_after = self.cells.t + self.dt
            for neuron_id in self.step():
                train.append((neuron_id, t_after))
        return train
