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

import math

import numpy as np


def _ragged_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate the integer ranges ``[starts[i] : starts[i]+counts[i])`` for all i.

    Vectorized gather of variable-length CSR slices (all fired cells' outgoing
    synapses in one shot). ``counts`` MUST be strictly positive — callers drop
    zero-fan-out sources first, otherwise a zero-length run corrupts the boundary
    write. Classic "cumsum of a difference array" trick: within a run the step is
    +1; at each run boundary the step jumps to the next start.
    """
    total = int(counts.sum())
    result = np.ones(total, dtype=np.int64)
    result[0] = starts[0]
    boundaries = np.cumsum(counts)[:-1]
    result[boundaries] += starts[1:] - starts[:-1] - counts[:-1]
    return np.cumsum(result)


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
    freeze learning too. Delayed delivery is a BUCKET RING drained by ``np.add.at``:
    each tick's arrivals are lexsorted by the SAME key the OOP heap compared —
    ``(arrival_time, target, effective_weight, source)`` — then accumulated in that
    order, which is what makes the float64 sums bit-exact (float addition is not
    associative, so summation ORDER is load-bearing). This replaced a per-spike heap
    that popped/pushed in the same order but was ~50x slower; equivalence is gated
    byte-for-byte against the heap build (commit 4d31553).
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = float(dt)
        self._cell_specs: list[tuple[int, dict]] = []
        self._cell_id_set: set[int] = set()  # O(1) duplicate detection (was O(N))
        self._built = False

        # Synapses are collected as ORDERED CHUNKS of parallel arrays, so the
        # singular and bulk APIs share one global insertion order. Singular calls
        # accumulate in a small buffer that is flushed to a chunk on the first
        # bulk call and at build; a bulk call stores its arrays directly (no
        # per-synapse Python tuples). Concatenating chunks in order, then a stable
        # sort by source index, makes a mixed singular/bulk build byte-identical
        # to a pure-singular one — which the heap tie-break, and therefore spike
        # delivery order, depends on.
        self._syn_chunks: list[tuple[np.ndarray, ...]] = []
        self._syn_singular: list[tuple[int, int, float, float, float, float]] = []

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
        self._syn_post_idx: np.ndarray | None = None   # (E,) int64 target array index
        self._syn_bucket_off: np.ndarray | None = None  # (E,) int64 max(1, ceil(delay))
        self._ring: list[list] = []                    # bucket-ring delivery slots
        self._ring_size: int = 0
        self._external: np.ndarray | None = None       # staged injections

    def add_cell(self, neuron_id: int, **params: float) -> None:
        if self._built:
            raise RuntimeError("cannot add cells after the network is built")
        if neuron_id in self._cell_id_set:  # O(1) set lookup, not an O(N) scan
            raise ValueError(f"duplicate neuron_id {neuron_id}")
        self._cell_id_set.add(neuron_id)
        self._cell_specs.append((neuron_id, params))

    def add_synapse(
        self, pre_id: int, post_id: int, weight: float, distance: float,
        propagation_speed: float = 1.0, decay_constant: float = 10.0,
    ) -> None:
        """Add one synapse. Identical behaviour/signature to before; buffered so
        it shares global insertion order with :meth:`add_synapses_bulk`."""
        if self._built:
            raise RuntimeError("cannot add synapses after the network is built")
        self._syn_singular.append(
            (pre_id, post_id, weight, distance, propagation_speed, decay_constant)
        )

    def _flush_singular(self) -> None:
        """Turn any buffered singular synapses into one chunk, in insertion order."""
        if not self._syn_singular:
            return
        specs = self._syn_singular
        self._syn_chunks.append((
            np.array([s[0] for s in specs], dtype=np.int64),   # pre_id
            np.array([s[1] for s in specs], dtype=np.int64),   # post_id
            np.array([s[2] for s in specs], dtype=np.float64),  # weight
            np.array([s[3] for s in specs], dtype=np.float64),  # distance
            np.array([s[4] for s in specs], dtype=np.float64),  # propagation_speed
            np.array([s[5] for s in specs], dtype=np.float64),  # decay_constant
        ))
        self._syn_singular = []

    def add_synapses_bulk(
        self, pre_ids, post_ids, weights, distances,
        propagation_speed=1.0, decay_constant=10.0,
    ) -> None:
        """Add many synapses at once from arrays/lists (scalars broadcast).

        Stores the arrays directly — no per-synapse Python tuples. Flushes the
        singular buffer first so the global insertion order (singular -> bulk ->
        singular) is preserved exactly.
        """
        if self._built:
            raise RuntimeError("cannot add synapses after the network is built")

        pre = np.asarray(pre_ids, dtype=np.int64).ravel()
        post = np.asarray(post_ids, dtype=np.int64).ravel()
        if pre.shape != post.shape:
            raise ValueError(
                f"pre_ids and post_ids must have the same length "
                f"({pre.size} vs {post.size})"
            )
        m = pre.size

        def bcast(value, name: str) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                return np.full(m, float(arr), dtype=np.float64)
            arr = arr.ravel()
            if arr.size != m:
                raise ValueError(
                    f"{name} must be a scalar or length {m}, got length {arr.size}"
                )
            return arr

        self._flush_singular()
        self._syn_chunks.append((
            pre, post,
            bcast(weights, "weights"),
            bcast(distances, "distances"),
            bcast(propagation_speed, "propagation_speed"),
            bcast(decay_constant, "decay_constant"),
        ))

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

        # Vectorized synapse construction. Concatenate all chunks IN ORDER, so the
        # arrays are in global insertion order (singular and bulk interleaved).
        self._flush_singular()
        ids_arr = np.asarray(ids, dtype=np.int64)  # sorted ascending == id2idx order
        if self._syn_chunks:
            pre_ids = np.concatenate([c[0] for c in self._syn_chunks])
            post_ids = np.concatenate([c[1] for c in self._syn_chunks])
            weights = np.concatenate([c[2] for c in self._syn_chunks])
            distances = np.concatenate([c[3] for c in self._syn_chunks])
            prop_speed = np.concatenate([c[4] for c in self._syn_chunks])
            decay = np.concatenate([c[5] for c in self._syn_chunks])
        else:
            pre_ids = post_ids = np.empty(0, dtype=np.int64)
            weights = distances = prop_speed = decay = np.empty(0, dtype=np.float64)

        # Validate against the sorted id array via searchsorted (not dict lookups
        # in a loop). searchsorted gives the array index of a registered id; a
        # position that is out of range, or whose id does not match, is unregistered.
        pre_idx = np.searchsorted(ids_arr, pre_ids)
        post_idx = np.searchsorted(ids_arr, post_ids)
        pre_ok = (pre_idx < n) & (ids_arr[np.minimum(pre_idx, n - 1)] == pre_ids)
        post_ok = (post_idx < n) & (ids_arr[np.minimum(post_idx, n - 1)] == post_ids)

        # Report the FIRST offender in insertion order, pre before post within an
        # edge (matching the original per-edge check order), verbatim message. E is
        # the "none" sentinel.
        e_count = pre_ids.size
        first_bad_pre = int(np.argmax(~pre_ok)) if not pre_ok.all() else e_count
        post_bad = pre_ok & ~post_ok  # only edges whose pre was fine
        first_bad_post = int(np.argmax(post_bad)) if post_bad.any() else e_count
        if first_bad_pre == e_count and first_bad_post == e_count:
            pass  # all edges reference registered cells
        elif first_bad_pre <= first_bad_post:
            raise ValueError(
                f"synapse pre_id {int(pre_ids[first_bad_pre])} is not a registered cell"
            )
        else:
            raise ValueError(
                f"synapse post_id {int(post_ids[first_bad_post])} is not a registered cell"
            )

        # Vectorized delay and effective weight. math.exp elementwise ON PURPOSE:
        # math.exp and np.exp can differ by 1 ULP, and bit-exactness against the
        # OOP layer is the whole value of this layer; exp is a small fraction of
        # build cost. IEEE division is identical between numpy and Python, so the
        # argument itself is safe to vectorize.
        delay = distances / prop_speed
        arg = -distances / decay
        eff_w = weights * np.fromiter(
            (math.exp(a) for a in arg), dtype=np.float64, count=arg.size
        )

        # Stable sort by source index -> CSR. Stable preserves insertion order for
        # equal sources, keeping heap tie-breaks (delivery order) identical to the
        # original singular build.
        order = np.argsort(pre_idx, kind="stable")
        self._syn_post_id = post_ids[order].astype(np.int32)
        self._syn_eff_w = eff_w[order]
        self._syn_delay = delay[order]
        indptr = np.zeros(n + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(pre_idx, minlength=n))
        self._syn_indptr = indptr

        # BUCKET-RING DELIVERY (replaces the per-spike heap; bit-exact, ~50x faster).
        #   * target ARRAY INDEX per synapse. id2idx is monotone (ids are sorted), so
        #     ordering by target_idx is identical to ordering by target_id — which is
        #     what the heap tie-break used. np.add.at then scatters in index-array
        #     order, the SAME order a sequential `raw[target] += w` would accumulate.
        #   * bucket offset = max(1, ceil(delay)). A cell pushes its bumps AFTER its
        #     own tick's drain, so the earliest an arrival can be delivered is the
        #     next tick (offset 1) even for a sub-1 ms delay — this reproduces the
        #     heap's "first tick_end > firing tick AND >= arrival_time" rule exactly.
        self._syn_post_idx = post_idx[order].astype(np.int64)
        off = np.ceil(self._syn_delay).astype(np.int64)
        np.maximum(off, 1, out=off)
        self._syn_bucket_off = off
        self._max_off = int(off.max()) if off.size else 1
        self._ring_size = self._max_off + 1
        # Each ring slot holds a list of (arrival_time, target_idx, eff_w, source_idx)
        # array-tuples appended during fan-out, concatenated + lexsorted at drain.
        self._ring = [[] for _ in range(self._ring_size)]

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
        new_t = cells.t + self.dt          # post-increment time (integer-valued)
        T = int(new_t)
        ring = self._ring
        rs = self._ring_size

        # (a) Drain this tick's bucket into the raw-input vector. Every arrival due
        #     at tick_end == new_t lives in ring[T % rs] (bucket == ceil(arrival),
        #     which the offset math guarantees). Concatenate the slot's fan-out
        #     batches, LEXSORT by (arrival_time, target_idx, eff_w, source_idx) so
        #     same-target arrivals are ordered exactly as the heap popped them, then
        #     np.add.at — which accumulates in index-array order, making the float64
        #     sums bit-identical to the sequential `raw[target] += w` it replaces.
        raw = self._external.copy()
        self._external[:] = 0.0
        b = T % rs
        bucket = ring[b]
        if bucket:
            if len(bucket) == 1:
                arr_t, tgt, w, src = bucket[0]
            else:
                arr_t = np.concatenate([e[0] for e in bucket])
                tgt = np.concatenate([e[1] for e in bucket])
                w = np.concatenate([e[2] for e in bucket])
                src = np.concatenate([e[3] for e in bucket])
            order = np.lexsort((src, w, tgt, arr_t))
            np.add.at(raw, tgt[order], w[order])
            ring[b] = []

        # (b) Vectorized cell dynamics.
        fired_idx = cells.step(raw)

        # (c) Fan-out: scatter every fired cell's outgoing bumps into future buckets,
        #     one vectorized gather across all fired cells (no per-spike Python loop).
        if fired_idx.size:
            indptr = self._syn_indptr
            counts = indptr[fired_idx + 1] - indptr[fired_idx]
            nz = counts > 0
            if nz.any():
                src_cells = fired_idx[nz]
                counts = counts[nz]
                syn = _ragged_ranges(indptr[src_cells], counts)
                tgt_all = self._syn_post_idx[syn]
                w_all = self._syn_eff_w[syn]
                arr_all = new_t + self._syn_delay[syn]
                src_all = np.repeat(src_cells, counts)
                dest = (T + self._syn_bucket_off[syn]) % rs
                for db in np.unique(dest):
                    m = dest == db
                    ring[db].append((arr_all[m], tgt_all[m], w_all[m], src_all[m]))

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
