"""[SOA] The vectorized compute layer must match the OOP oracle BIT-FOR-BIT.

Matching comes first; performance is secondary and lives in test_soa_perf-style
assertions only after equivalence is proven. The OOP ``Cell`` / ``Network`` are
the oracle; ``SoANetwork`` is validated against them here.

Phase 1 is DYNAMICS ONLY — no STDP/CST/decay. Any OOP network used as a reference
therefore FREEZES learning (``learning_rate=0`` and ``tau_decay=inf``), so its
synapse weights never move and only the dynamics are compared.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from phoenix.cell import Cell
from phoenix.network_graph import Network
from phoenix.soa import CellArrays, SoANetwork
from phoenix.synapse import Synapse


# ---------------------------------------------------------------------------
# S1. Single cell, input that coincides with refractory — the 4 details at once
# ---------------------------------------------------------------------------
def test_soa_single_cell_matches_oop() -> None:
    """20 ticks, continuous input of 30 mV (fires, then gets rejected in refractory).

    This exercises all four critical details: the integrate order, the
    post-increment firing clock, Option-A rejection on the pre-increment clock,
    and the two-different-`t` refractory checks. Vm and spike times must match to
    < 1e-9.
    """
    tau, weight = 20.0, 30.0

    oop = Cell(neuron_id=0, tau=tau)
    oop_vm, oop_spikes = [], []
    for _ in range(20):
        oop.receive_input(weight)            # at pre-increment t
        spike = oop.integrate(1.0)
        oop_vm.append(oop.Vm)
        if spike is not None:
            oop_spikes.append(spike.timestamp)

    soa = CellArrays(1, tau=tau)
    soa_vm, soa_spikes = [], []
    for _ in range(20):
        fired = soa.step(np.array([weight], dtype=np.float64))
        soa_vm.append(float(soa.Vm[0]))
        if fired.size:
            soa_spikes.append(soa.t)

    assert oop_spikes  # it really did fire (and get rejected in between)
    assert soa_spikes == oop_spikes
    for a, b in zip(soa_vm, oop_vm):
        assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# S2. Option A: input arriving during refractory is DROPPED (the biggest trap)
# ---------------------------------------------------------------------------
def test_soa_refractory_input_rejection() -> None:
    """Fire, then deliver input while refractory. It must vanish, not defer.

    Verified against OOP tick by tick: the rejected input leaves no trace — Vm
    follows pure leak during refractory, identical in both.
    """
    tau, refractory_period = 20.0, 2.0

    oop = Cell(neuron_id=0, tau=tau, refractory_period=refractory_period)
    soa = CellArrays(1, tau=tau, refractory_period=refractory_period)

    # Tick 1: a big kick fires the cell in both.
    oop.receive_input(30.0)
    oop_fired_1 = oop.integrate(1.0) is not None
    soa_fired_1 = soa.step(np.array([30.0])).size > 0
    assert oop_fired_1 and soa_fired_1
    assert oop.refractory_until == soa.refractory_until[0] == 3.0  # t(=1)+2

    # Ticks 2-3: deliver input WHILE refractory (t=1<3, then t=2<3). Dropped.
    for _ in range(2):
        oop.receive_input(30.0)              # rejected: t < refractory_until
        oop_spike = oop.integrate(1.0)
        soa_fired = soa.step(np.array([30.0]))
        assert oop_spike is None
        assert soa_fired.size == 0
        # The dropped input left NO trace: Vm is pure leak from Vreset, matching.
        assert abs(soa.Vm[0] - oop.Vm) < 1e-9
        assert soa.Vm[0] == soa.Vreset[0]    # stayed at reset (leak from rest = id)


# ---------------------------------------------------------------------------
# S3. 50 cells, varied tau, dense random input — full spike sequence identical
# ---------------------------------------------------------------------------
def test_soa_multi_cell_spike_sequence() -> None:
    n_cells, ticks = 50, 80
    rng = random.Random(20260717)
    taus = [rng.uniform(5.0, 30.0) for _ in range(n_cells)]

    # A fixed dense input matrix, fed IDENTICALLY to both implementations.
    inputs = [
        [rng.uniform(0.0, 14.0) if rng.random() < 0.6 else 0.0 for _ in range(n_cells)]
        for _ in range(ticks)
    ]

    # OOP oracle: 50 independent Cells driven tick by tick.
    oop_cells = [Cell(neuron_id=i, tau=taus[i]) for i in range(n_cells)]
    oop_train: list[tuple[int, float]] = []
    for tick in range(ticks):
        for i, cell in enumerate(oop_cells):
            if inputs[tick][i]:
                cell.receive_input(inputs[tick][i])
            spike = cell.integrate(1.0)
            if spike is not None:
                oop_train.append((i, spike.timestamp))

    # SoA: one vectorized CellArrays.
    soa = CellArrays(n_cells, tau=np.array(taus))
    soa_train: list[tuple[int, float]] = []
    for tick in range(ticks):
        fired = soa.step(np.array(inputs[tick], dtype=np.float64))
        for idx in fired:
            soa_train.append((int(idx), soa.t))

    assert len(oop_train) > 300                # a real, active workload (~hundreds)
    assert soa_train == oop_train              # EXACT sequence, not "within tolerance"


# ---------------------------------------------------------------------------
# S4. Small network with synapses + delays — delivery and coincidence match
# ---------------------------------------------------------------------------
def _frozen_syn(pre: int, post: int, weight: float, delay: float,
                decay: float = 1000.0) -> Synapse:
    """A synapse with learning TRULY frozen: weights never move (phase-1 dynamics only).

    Freezing is three things, not two. learning_rate=0 stops STDP and tau_decay=inf
    stops passive decay, but the weight BOUNDS are still enforced at runtime:
    apply_decay applies max(w_min, .) on every post-spike (default w_min=0 floors any
    negative weight to 0), and update_weight clamps to [w_min, w_max]. So a faithful
    oracle for signed / heavy-tailed weights MUST also lift the bounds — otherwise it
    silently deletes inhibition and caps the tail. See soa.py module docstring.
    """
    return Synapse(
        pre_id=pre, post_id=post, weight=weight, distance=delay,
        propagation_speed=1.0, decay_constant=decay,
        learning_rate=0.0, tau_decay=math.inf,
        w_min=-math.inf, w_max=math.inf,
    )


def test_soa_network_with_delays() -> None:
    # A convergent unit: A,B -> C with delays that make the two bumps COINCIDE at
    # C, plus a plain relay C -> D. 15 mV each is sub-threshold alone; together
    # (30 mV) they fire C.
    def build_oop() -> Network:
        net = Network(dt=1.0)
        for nid in (1, 2, 3, 4):
            net.add_cell(Cell(neuron_id=nid))
        net.add_synapse(_frozen_syn(1, 3, 15.0, 1.0))
        net.add_synapse(_frozen_syn(2, 3, 15.0, 2.0))
        net.add_synapse(_frozen_syn(3, 4, 30.0, 1.0))
        return net

    def build_soa() -> SoANetwork:
        net = SoANetwork(dt=1.0)
        for nid in (1, 2, 3, 4):
            net.add_cell(nid)
        net.add_synapse(1, 3, 15.0, 1.0, decay_constant=1000.0)
        net.add_synapse(2, 3, 15.0, 2.0, decay_constant=1000.0)
        net.add_synapse(3, 4, 30.0, 1.0, decay_constant=1000.0)
        return net

    oop, soa = build_oop(), build_soa()

    # Fire A and B on the same tick so their staggered delays converge at C.
    oop_train: list[tuple[int, float]] = []
    soa_train: list[tuple[int, float]] = []
    for tick in range(60):
        if tick == 0:
            oop.inject(1, 100.0)
            oop.inject(2, 100.0)
            soa.inject(1, 100.0)
            soa.inject(2, 100.0)
        oop_train.extend((s.neuron_id, s.timestamp) for s in oop.step())
        t_after = soa.cells.t + soa.dt
        soa_train.extend((nid, t_after) for nid in soa.step())

    assert any(nid == 4 for nid, _ in oop_train)  # the wave really propagated A/B->C->D
    assert soa_train == oop_train


def test_soa_bucket_ring_scatter_uses_target_index_not_id() -> None:
    """Regression: ring delivery must scatter to the target's ARRAY INDEX, not its id.

    The bucket ring drains with ``np.add.at(raw, target_index, weight)``. If the
    target NEURON ID is used as the index instead, a grid where id == index hides
    it, but any id != index misroutes the bump. Here ids are widely spaced
    (100,200,300,400) so id != index strongly, over the same sparse convergent-delay
    construction as S4 (which is in the regime where SoA is bit-exact vs OOP): A,B
    stagger-converge on C, C relays to D. A single misrouted bump changes the train.
    """
    id_a, id_b, id_c, id_d = 100, 200, 300, 400  # sorted -> indices 0,1,2,3 (id != index)

    def build_oop() -> Network:
        net = Network(dt=1.0)
        for nid in (id_a, id_b, id_c, id_d):
            net.add_cell(Cell(neuron_id=nid))
        net.add_synapse(_frozen_syn(id_a, id_c, 15.0, 1.0))
        net.add_synapse(_frozen_syn(id_b, id_c, 15.0, 2.0))
        net.add_synapse(_frozen_syn(id_c, id_d, 30.0, 1.0))
        return net

    def build_soa() -> SoANetwork:
        net = SoANetwork(dt=1.0)
        for nid in (id_a, id_b, id_c, id_d):
            net.add_cell(nid)
        net.add_synapse(id_a, id_c, 15.0, 1.0, decay_constant=1000.0)
        net.add_synapse(id_b, id_c, 15.0, 2.0, decay_constant=1000.0)
        net.add_synapse(id_c, id_d, 30.0, 1.0, decay_constant=1000.0)
        return net

    oop, soa = build_oop(), build_soa()
    oop_train: list[tuple[int, float]] = []
    soa_train: list[tuple[int, float]] = []
    for tick in range(60):
        if tick == 0:
            oop.inject(id_a, 100.0); oop.inject(id_b, 100.0)
            soa.inject(id_a, 100.0); soa.inject(id_b, 100.0)
        oop_train.extend((s.neuron_id, s.timestamp) for s in oop.step())
        t_after = soa.cells.t + soa.dt
        soa_train.extend((nid, t_after) for nid in soa.step())

    assert any(nid == id_d for nid, _ in oop_train)  # the wave reached D via C
    assert soa_train == oop_train                    # EXACT — misrouted bump would break it


def test_soa_matches_oop_with_inhibition_and_heavy_tail() -> None:
    """Bit-exact vs OOP in the EXPERIMENTS' regime: E/I balance + heavy-tailed weights.

    This is the regime the capacity runs use, and the one where a NAIVELY frozen OOP
    reference (learning_rate=0, tau_decay=inf, but default bounds [0, 20]) diverges:
    apply_decay's max(w_min=0, .) floors every inhibitory weight to 0 the first time
    its post cell fires, and w_max=20 caps the heavy tail. _frozen_syn lifts the
    bounds, so SoA (which never touches weights) and OOP match BIT-FOR-BIT. If this
    ever diverges, a weight bound has crept back into the oracle. See soa.py docstring.
    """
    rng = random.Random(11)
    n = 60
    inh = {i for i in range(n) if rng.random() < 0.2}          # ~20% inhibitory

    oop, soa = Network(dt=1.0), SoANetwork(dt=1.0)
    for i in range(n):
        oop.add_cell(Cell(neuron_id=i, tau=3.0, refractory_period=2.0))
        soa.add_cell(i, tau=3.0, refractory_period=2.0)
    for pre in range(n):
        for _ in range(8):                                      # fan-out 8, recurrent
            post = rng.randrange(n)
            w = math.exp(rng.gauss(math.log(6.98) - 2.0, 2.0))  # log-normal, heavy tail (>20 occurs)
            if pre in inh:
                w *= -4.0                                       # inhibitory: NEGATIVE weight
            dist = rng.uniform(1.0, 8.0)
            oop.add_synapse(_frozen_syn(pre, post, w, dist, decay=20.0))
            soa.add_synapse(pre, post, w, dist, decay_constant=20.0)

    # sanity: the regime actually contains what would trip the bounds
    has_neg = any(s.weight < 0 for outs in oop.outgoing.values() for s in outs)
    has_heavy = any(s.weight > 20 for outs in oop.outgoing.values() for s in outs)
    assert has_neg and has_heavy

    drive = random.Random(5)
    oop_train, soa_train = [], []
    for _ in range(200):
        for i in range(n):
            if drive.random() < 0.02:
                oop.inject(i, 30.0)
                soa.inject(i, 30.0)
        oop_train.extend((s.neuron_id, s.timestamp) for s in oop.step())
        t_after = soa.cells.t + soa.dt
        soa_train.extend((nid, t_after) for nid in soa.step())

    assert len(oop_train) > 200                       # genuinely active recurrent E/I net
    assert soa_train == oop_train                     # BIT-EXACT with bounds lifted


# ---------------------------------------------------------------------------
# S5. dt-invariance / silent-cell exactness carries over to SoA
# ---------------------------------------------------------------------------
def test_soa_dt_invariance() -> None:
    """A cell silent for 1,000 ticks then driven behaves identically to the oracle.

    SoA integrates every cell every tick (the leak from Vrest is the identity, so
    it is cheap and exact), so this pins that a long silence leaves no residue.
    """
    silent = 1_000

    oop = Cell(neuron_id=0, tau=20.0)
    soa = CellArrays(1, tau=20.0)

    # Both sit silent for the same number of ticks.
    for _ in range(silent):
        oop.integrate(1.0)
        soa.step(np.array([0.0]))

    # Bit-identical rest state after the silence (leak from Vrest is the identity).
    assert soa.t == oop.t
    assert soa.Vm[0] == oop.Vm

    # Now drive both identically, on the same tick.
    oop.receive_input(100.0)
    oop_spike = oop.integrate(1.0)
    fired = soa.step(np.array([100.0]))

    assert oop_spike is not None and fired.size == 1
    assert soa.t == oop_spike.timestamp
    assert soa.Vm[0] == oop.Vm
    assert soa.last_spike_time[0] == oop.last_spike_time


# ---------------------------------------------------------------------------
# S6. THE GOAL: temporal-order signature at scale (>=1000 cells)
# ---------------------------------------------------------------------------
def test_soa_state_space_signature() -> None:
    """A >=1000-cell delay-line network separates a sequence from its reverse.

    This is the central claim ("ktb" vs "btk") demonstrated at scale, with PURE
    DYNAMICS and no learning. 4 shared input cells encode a 4-symbol sequence.
    Each detector fires only on true COINCIDENCE of its fan-in bumps: forward
    detectors have delays (SEQLEN-i)*hop so a forward-order input makes all four
    bumps land on one tick; reverse detectors mirror them. Detector tau is short
    (5 ms) so a wrong-order, spread-out arrival LEAKS AWAY before it can sum past
    threshold — sharp coincidence detection, which is the whole mechanism.
    """
    seq_len, hop, weight, det_tau = 4, 2, 9.0, 5.0
    n_detectors = 500  # per bank -> 4 + 500 + 500 = 1004 cells

    def build() -> tuple[SoANetwork, list[int], list[int]]:
        net = SoANetwork(dt=1.0)
        for i in range(seq_len):
            net.add_cell(i)  # inputs: default tau
        forward = list(range(seq_len, seq_len + n_detectors))
        reverse = list(range(seq_len + n_detectors, seq_len + 2 * n_detectors))
        for det in forward:
            net.add_cell(det, tau=det_tau)
            for i in range(seq_len):
                net.add_synapse(i, det, weight, (seq_len - i) * hop, decay_constant=1e9)
        for det in reverse:
            net.add_cell(det, tau=det_tau)
            for i in range(seq_len):
                net.add_synapse(i, det, weight, (i + 1) * hop, decay_constant=1e9)
        return net, forward, reverse

    total = seq_len + 2 * n_detectors
    assert total >= 1000

    def signature(order: list[int]) -> tuple[np.ndarray, int, int]:
        net, forward, reverse = build()
        counts: dict[int, int] = {}
        for tick in range(40):
            for position, symbol in enumerate(order):
                if tick == position * hop:  # sequential presentation
                    net.inject(symbol, 100.0)
            for nid in net.step():
                counts[nid] = counts.get(nid, 0) + 1
        vec = np.array([counts.get(n, 0) for n in range(total)], dtype=np.float64)
        fired_forward = sum(1 for d in forward if counts.get(d, 0))
        fired_reverse = sum(1 for d in reverse if counts.get(d, 0))
        return vec, fired_forward, fired_reverse

    fwd_vec, ff, fr = signature([0, 1, 2, 3])   # "ktb"  (forward)
    rev_vec, rf, rr = signature([3, 2, 1, 0])   # "btk"  (reversed)

    # The temporal order is read out cleanly, and ONLY by coincidence.
    assert ff == n_detectors and fr == 0        # forward input -> only forward bank
    assert rr == n_detectors and rf == 0        # reverse input -> only reverse bank

    # A non-zero state-space separation — the wall this opens the door to testing.
    distance = float(np.linalg.norm(fwd_vec - rev_vec))
    assert distance > 0.0
    assert distance == pytest.approx(math.sqrt(2 * n_detectors), abs=1e-9)


# ---------------------------------------------------------------------------
# S7. Performance & memory — matching is proven, so now the point of SoA
# ---------------------------------------------------------------------------
def test_soa_memory_footprint() -> None:
    """The SoA footprint is what makes the N-cell scale reachable.

    Deterministic (dtype sizes), so this is a hard assertion, not a measurement.
    Cell state is 9 float64 arrays (72 B/cell); a synapse edge is int32 + two
    float64 (20 B) plus an amortized CSR offset. Versus the OOP objects (628 and
    792 B with __slots__), this projects 1M cells + 10M synapses to ~0.27 GB —
    down from 8.5 GB, the wall this whole step exists to remove.
    """
    net = SoANetwork(dt=1.0)
    for i in range(200):
        net.add_cell(i)
    for i in range(200):
        for k in range(3):
            net.add_synapse(i, (i + k + 1) % 200, 8.0, 2.0)
    net._build()

    cell_arrays = [
        net.cells.Vrest, net.cells.Vthresh, net.cells.Vreset, net.cells.tau,
        net.cells.refractory_period, net.cells._leak, net.cells.Vm,
        net.cells.refractory_until, net.cells.last_spike_time,
    ]
    cell_bytes = sum(a.itemsize for a in cell_arrays)
    edge_bytes = (
        net._syn_post_id.itemsize + net._syn_eff_w.itemsize + net._syn_delay.itemsize
    )

    assert cell_bytes == 72          # 9 x float64
    assert edge_bytes == 20          # int32 + 2 x float64
    assert all(a.dtype == np.float64 for a in cell_arrays)  # float64 only, no drift

    projected_gb = (cell_bytes * 1_000_000 + edge_bytes * 10_000_000) / 1e9
    assert projected_gb < 1.0        # ~0.27 GB; was 8.5 GB with OOP __slots__


def test_soa_throughput_is_vectorized() -> None:
    """A LOOSE floor that catches accidental de-vectorization (a Python per-cell
    loop would be ~100x slower). Measured ~27M cell-updates/s; asserting >2M
    leaves a wide margin so this is not flaky, while still failing hard if the
    vectorized step is ever replaced by a scalar loop.
    """
    import time

    n = 200_000
    rng = np.random.default_rng(0)
    cells = CellArrays(n, tau=rng.uniform(5.0, 30.0, n))
    steps = 20

    start = time.perf_counter()
    for _ in range(steps):
        raw = np.where(rng.random(n) < 0.02, 40.0, 0.0)
        cells.step(raw)
    elapsed = time.perf_counter() - start

    updates_per_sec = n * steps / elapsed
    assert updates_per_sec > 2_000_000
