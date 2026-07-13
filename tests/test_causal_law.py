"""The Causal Discrimination Law + non-linear lazy decay.

>>> THE LAW <<<

    post_firing_rate * verify_window  <<  1        (i.e. ISI >> verify_window)

If the post cell fires in nearly EVERY verification window, a noise spike and a
causal spike produce IDENTICAL observations. The causal information is NOT
PRESENT IN THE DATA, and no local, observational measure can recover it. This is
an information-theoretic limit, not a tuning issue.

Investigating the idle-synapse gap uncovered that CST works in the FRAGILE
topology and FAILS in the saturated one: at a 100 Hz post rate the noise synapse
out-competes the real assembly and pins at w_max. This file pins that failure
(L1) so it can never regress into looking "fine", and pins the sparse regime
that fixes it (L2).

Three pieces, none of which works alone:
  1. Reverberation needs an ASSEMBLY, not a 2-cell loop (L3 vs L4).
  2. The firing rate is set by TOPOLOGY (cycle period), not homeostasis.
  3. Non-linear LAZY DECAY prunes the idle synapse (L5-L7) — the [OPEN-1] gap.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from phoenix.cell import Cell
from phoenix.network_graph import Network, assembly_ignition_voltage
from phoenix.synapse import Synapse

RING = 10          # cells in the assembly
FAN_IN = 3         # convergent inputs per cell (3 x 11 mV = 33 mV > the 25 mV gap)
W0 = 11.0
NOISE_ID = 100
IDLE_ID = 101


def _syn(pre: int, post: int, weight: float, delay: float) -> Synapse:
    return Synapse(
        pre_id=pre, post_id=post, weight=weight, distance=delay,
        propagation_speed=1.0, decay_constant=1000.0,
    )


def _ring(hop: float, *, noise_rate: float = 0.0, seed: int = 42,
          run_ms: int = 30_000, with_idle: bool = False) -> tuple[Network, list]:
    """A `RING`-cell reverberating assembly whose firing rate is set by `hop`.

    Cell i is driven by its 3 predecessors, with delays hop / 2*hop / 3*hop so all
    three bumps ARRIVE SIMULTANEOUSLY and sum past threshold. Activity therefore
    advances one cell every `hop` ms, so each cell fires once per lap:

        post_rate = 1000 / (RING * hop) Hz

    hop=1 -> 100 Hz (saturated: the Law is violated)
    hop=3 ->  33 Hz (sparse:    the Law is satisfied)
    """
    net = Network(dt=1.0)
    for i in range(RING):
        net.add_cell(Cell(neuron_id=i))
    net.add_cell(Cell(neuron_id=NOISE_ID))
    if with_idle:
        net.add_cell(Cell(neuron_id=IDLE_ID))

    for i in range(RING):
        for k in range(1, FAN_IN + 1):
            net.add_synapse(_syn((i - k) % RING, i, W0, k * hop))
    net.add_synapse(_syn(NOISE_ID, 0, W0, 1.0))     # active noise -> cell 0
    if with_idle:
        net.add_synapse(_syn(IDLE_ID, 1, W0, 1.0))  # its pre cell NEVER fires

    rng = random.Random(seed)
    spikes: list = []
    for _ in range(run_ms):
        for i in range(FAN_IN):          # ignition only: cells 0,1,2 at 0,hop,2*hop
            if net.current_time == i * hop:
                net.inject(i, 100.0)
        if noise_rate and rng.random() < noise_rate:
            net.inject(NOISE_ID, 100.0)
        spikes.extend(net.step())
    return net, spikes


def _assembly(net: Network) -> list[Synapse]:
    return [s for i in range(RING) for s in net.incoming[i] if s.pre_id < RING]


def _noise(net: Network) -> Synapse:
    return next(s for s in net.incoming[0] if s.pre_id == NOISE_ID)


def _cell_rate(spikes: list, neuron_id: int, since: float, until: float) -> float:
    n = sum(1 for s in spikes if s.neuron_id == neuron_id and since < s.timestamp <= until)
    return n / ((until - since) / 1000.0)


# ---------------------------------------------------------------------------
# L1. THE LAW, pinned as a FAILURE — the most important test in this file
# ---------------------------------------------------------------------------
def test_saturated_post_rate_destroys_discrimination() -> None:
    """THIS TEST ASSERTS A FAILURE ON PURPOSE. It must never start "passing".

    hop=1 -> the post cell fires every 10 ms -> post_rate * verify_window = 1.0.
    The post fires in EVERY verification window, so a noise spike and a causal
    spike produce identical observations: the causal information is not in the
    data. No local measure can recover it — we also tried a baseline-lift measure
    P(post|pre) - P(post), which correctly returned exactly 0.0 for everything,
    because P(post) = 1.0.

    The consequence is a catastrophe that would be undiagnosable at N-cell scale:
    the NOISE synapse out-competes the true assembly and pins at w_max, silently
    amplifying noise while starving the real signal.

    If this test ever reports success, the Law has been violated and the
    discrimination is a mirage.
    """
    net, spikes = _ring(hop=1, noise_rate=0.02, run_ms=100_000)

    post_rate = _cell_rate(spikes, 0, 95_000, 100_000)
    noise = _noise(net)
    assembly_mean = statistics.mean(s.weight for s in _assembly(net))

    # The Law's precondition is VIOLATED: the post fires in every window.
    assert post_rate == pytest.approx(100.0, abs=2.0)
    assert post_rate * noise.verify_window / 1000.0 >= 1.0

    # And so the discrimination collapses — assert the failure explicitly.
    assert noise.causal_success > 0.8                      # noise looks causal
    assert noise.weight == pytest.approx(noise.w_max, abs=1e-3)  # pinned at w_max
    assert noise.weight > assembly_mean                    # it BEATS the real thing


# ---------------------------------------------------------------------------
# L2. Same code, same noise — only the RATE changed
# ---------------------------------------------------------------------------
def test_sparse_post_rate_restores_discrimination() -> None:
    """hop=3 -> 33 Hz -> post_rate * verify_window = 0.33. The Law is satisfied.

    Nothing else differs from L1: same topology, same noise, same seed, same
    learning rules. Only the CYCLE PERIOD changed. That is the whole point — the
    rate is set by TOPOLOGY, and it is the rate that decides whether causal
    learning is possible at all.
    """
    net, spikes = _ring(hop=3, noise_rate=0.02, run_ms=30_000)

    post_rate = _cell_rate(spikes, 0, 25_000, 30_000)
    noise = _noise(net)
    assembly = _assembly(net)
    assembly_mean = statistics.mean(s.weight for s in assembly)
    assembly_cs = statistics.mean(s.causal_success for s in assembly)

    # The Law's precondition HOLDS.
    assert post_rate == pytest.approx(33.0, abs=2.0)
    assert post_rate * noise.verify_window / 1000.0 < 0.5

    # ...and discrimination is restored.
    assert noise.causal_success < 0.5 < assembly_cs
    assert noise.weight < assembly_mean
    assert abs(noise.weight - noise.w_max) > 1.0  # nowhere near the ceiling


# ---------------------------------------------------------------------------
# L3. Why an ASSEMBLY is required: a 2-cell loop is killed by ONE spike
# ---------------------------------------------------------------------------
def test_two_cell_loop_dies_to_a_single_spike_in_a_vulnerable_phase() -> None:
    """A single stray spike ends a 2-cell loop permanently — but ONLY in a
    vulnerable phase.

    Mechanism: the intruder makes the post cell fire EARLY; that cell enters
    refractory having already spent its spike; the real loop bump then arrives
    DURING refractory and is hard-rejected (Option A); the chain is broken, and
    nothing re-ignites it.

    MEASURED CORRECTION: the kill is PHASE-DEPENDENT, not universal. Scanning the
    loop's 4 ms cycle, one spike kills it in exactly 5 of 20 phases (~25%) —
    precisely those landing in the vulnerable quarter; at other phases the loop
    shrugs it off and keeps firing. This test pins a killing phase (t=100)
    deliberately.

    The fragility conclusion still holds — under SUSTAINED noise a
    vulnerable-phase strike becomes inevitable, and the loop has no way to
    re-ignite — but "one spike ALWAYS kills it" is FALSE, and no argument should
    rely on it. This is why assemblies (L4), not 2-cell loops, are the unit of
    reverberation.
    """
    hit_at = 100.0
    net = Network(dt=1.0)
    for neuron_id in (1, 2, 99):
        net.add_cell(Cell(neuron_id=neuron_id))
    for pre, post in ((1, 2), (2, 1)):
        net.add_synapse(_syn(pre, post, 15.0, 1.0))
        net.add_synapse(_syn(pre, post, 15.0, 2.0))
    net.add_synapse(_syn(99, 2, 15.0, 1.0))

    net.inject(1, 100.0)  # ignite
    spikes: list = []
    for _ in range(400):
        if net.current_time == hit_at:
            net.inject(99, 100.0)  # EXACTLY ONE stray spike
        spikes.extend(net.step())

    loop_spikes = [s for s in spikes if s.neuron_id in (1, 2)]
    before = [s for s in loop_spikes if s.timestamp < hit_at]
    after = [s for s in loop_spikes if s.timestamp > hit_at + 15]

    assert len(before) > 20      # it was healthily reverberating...
    assert len(after) == 0       # ...and ONE spike ended it, permanently.


# ---------------------------------------------------------------------------
# L4. The assembly shrugs it off — robustness is emergent, not engineered
# ---------------------------------------------------------------------------
def test_assembly_survives_repeated_hits() -> None:
    """Contrast with L3: the SAME noise that permanently kills a 2-cell loop
    leaves a 10-cell assembly running at full rate.

    Robustness here is an EMERGENT STATISTICAL PROPERTY OF REDUNDANCY: the
    assembly distributes the activity across many cells, so corrupting one does
    not lose the pattern. It is not a rescue mechanism bolted on — and no rescue
    mechanism was added (spontaneous activity cannot fire a spike by construction,
    and partial refractory was measured to have zero effect).
    """
    net, spikes = _ring(hop=3, noise_rate=0.10, seed=7, run_ms=30_000)

    tail_rate = _cell_rate(spikes, 0, 25_000, 30_000)

    # Under CONTINUOUS 10% noise the assembly is still reverberating, on rate.
    assert tail_rate > 3.0
    assert tail_rate == pytest.approx(33.0, abs=5.0)


# ---------------------------------------------------------------------------
# L5. [OPEN-1] CLOSED: the idle synapse is finally pruned
# ---------------------------------------------------------------------------
def test_idle_synapse_is_pruned() -> None:
    """A synapse whose presynaptic cell NEVER fires used to live forever.

    With hits = misses = 0 its causal_success is None — it has no opinion, so the
    gating had nothing to act on and the weight sat at its initial value
    indefinitely, contributing dead charge to its post cell's fan-in. In a large
    network most synapses are idle, so this accumulates.

    Lazy decay closes it: an idle synapse is treated as causal_success = 0.0 and
    therefore decays at the FULL, ungated rate. This is the ONLY mechanism in the
    system that prunes a synapse that never fires.
    """
    net, _ = _ring(hop=3, noise_rate=0.02, run_ms=100_000, with_idle=True)

    idle = next(s for s in net.incoming[1] if s.pre_id == IDLE_ID)
    assembly = _assembly(net)

    # It never fired, so it never had an opinion about its own causality...
    assert idle.hits == 0 and idle.misses == 0
    assert idle.causal_success is None

    # ...and it has been pruned away regardless (11.0 -> ~0.07).
    assert idle.weight < 2.0
    assert idle.weight < 0.5 * W0

    # While the PROVEN assembly is untouched — decay is gated, not blind.
    assert all(s.weight > 8.0 for s in assembly)


# ---------------------------------------------------------------------------
# L6. Why QUADRATIC: linear gating leaves a permanent bleed
# ---------------------------------------------------------------------------
def test_quadratic_gating_protects_the_causal_synapse() -> None:
    """Derived from the formula, not hardcoded.

    leak = (1 - cs) ** decay_power / tau_decay

    A proven loop's causal_success saturates at ~0.996, never 1.0, because of the
    Bayesian discount n/(n + n0_cs). Under LINEAR gating that residual 0.4% is a
    permanent bleed that killed the loop depending on the RNG seed. Squaring it
    makes the leak ~250x smaller, while leaving an IDLE synapse (cs = 0) decaying
    at the full, ungated rate — exactly the asymmetry we need.
    """
    elapsed = 100_000.0  # 100 s
    tau_decay = 20_000.0

    proven = Synapse(pre_id=1, post_id=2, weight=10.0, distance=1.0,
                     tau_decay=tau_decay, decay_power=2.0)
    idle = Synapse(pre_id=3, post_id=2, weight=10.0, distance=1.0,
                   tau_decay=tau_decay, decay_power=2.0)

    # Give `proven` a causal record of cs ~= 0.996 (hits, no misses).
    hits = 1200
    proven.hits = hits
    cs = (hits / hits) * (hits / (hits + proven.n0_cs))
    assert proven.causal_success == pytest.approx(cs)
    assert cs == pytest.approx(0.996, abs=0.002)

    proven.apply_decay(elapsed)
    idle.apply_decay(elapsed)  # cs is None -> treated as 0.0

    expected_proven = 10.0 * math.exp(-(((1 - cs) ** 2) / tau_decay) * elapsed)
    expected_idle = 10.0 * math.exp(-((1.0 ** 2) / tau_decay) * elapsed)
    assert proven.weight == pytest.approx(expected_proven)
    assert idle.weight == pytest.approx(expected_idle)

    # The causal synapse keeps essentially all of its weight...
    proven_loss = 1.0 - proven.weight / 10.0
    assert proven_loss < 0.001  # < 0.1%

    # ...while the idle one is annihilated.
    idle_loss = 1.0 - idle.weight / 10.0
    assert idle_loss > 0.9  # > 90%

    # Quadratic is dramatically gentler on the proven synapse than linear.
    linear_leak = (1 - cs) ** 1.0
    quadratic_leak = (1 - cs) ** 2.0
    assert linear_leak / quadratic_leak > 200


# ---------------------------------------------------------------------------
# L7. Lazy == exact: skipping ticks is safe
# ---------------------------------------------------------------------------
def test_decay_is_lazy_not_per_tick() -> None:
    """One long step must equal many small ones, or the event-driven model is unsound.

    Decay is applied ON DEMAND from elapsed time, never swept over every synapse
    every tick (which would be O(synapses x ticks) and destroy the neuromorphic
    cost model). That is only legitimate if the elapsed-time formulation is exact.
    It is, because exp(-a) * exp(-b) == exp(-(a + b)).
    """
    def fresh() -> Synapse:
        s = Synapse(pre_id=1, post_id=2, weight=10.0, distance=1.0)
        s.hits, s.misses = 8, 2  # a partial, mid-range causal_success
        return s

    one_shot = fresh()
    one_shot.apply_decay(5_000.0)

    incremental = fresh()
    for t in range(1, 5_001):
        incremental.apply_decay(float(t))

    assert one_shot.weight == pytest.approx(incremental.weight, abs=1e-9)

    # And decay never runs backwards.
    frozen = one_shot.weight
    one_shot.apply_decay(4_000.0)  # an earlier time: a no-op
    assert one_shot.weight == frozen


# ---------------------------------------------------------------------------
# L8. The full stack, on the verified configuration
# ---------------------------------------------------------------------------
def test_full_stack_is_robust() -> None:
    """10 cells, fan-in 3, hop 3, w=11, tau_decay=20000, 100 s, fixed seed.

    Everything at once: the Law satisfied by topology, the assembly reverberating,
    the active-noise synapse annihilated, and the idle synapse pruned.
    """
    net, spikes = _ring(hop=3, noise_rate=0.02, seed=42,
                        run_ms=100_000, with_idle=True)

    tail_rate = _cell_rate(spikes, 0, 95_000, 100_000)
    assembly = _assembly(net)
    assembly_mean = statistics.mean(s.weight for s in assembly)
    noise = _noise(net)
    idle = next(s for s in net.incoming[1] if s.pre_id == IDLE_ID)

    # 1) Reverberation is alive, and inside the Law.
    assert tail_rate > 3.0
    assert tail_rate == pytest.approx(33.4, abs=2.0)
    assert tail_rate * noise.verify_window / 1000.0 < 0.5

    # 2) The assembly's weights are healthy and interior.
    assert assembly_mean == pytest.approx(10.8, abs=1.0)
    assert all(0.0 < s.weight < s.w_max for s in assembly)
    assert statistics.mean(s.causal_success for s in assembly) > 0.9

    # 3) The ACTIVE noise synapse is annihilated (11.0 -> ~0.001).
    assert noise.weight < 0.5
    assert noise.causal_success < 0.5

    # 4) The IDLE synapse is pruned (11.0 -> ~0.07). [OPEN-1] closed.
    assert idle.weight < 2.0
    assert idle.causal_success is None


# ---------------------------------------------------------------------------
# B1/B2. The horizon boundary belongs to the HIT (the [CD-2] bug fix)
# ---------------------------------------------------------------------------
def _drive_at_latency(synapse: Synapse, latency: float, n_trials: int) -> None:
    """Drive a pre/post pair, replicating Network.step()'s REAL call order.

    Network.step() runs resolve_timeouts (step d2) BEFORE on_post_spike (step e),
    every tick. Reproducing that order is the whole point: it is what exposed the
    boundary bug.
    """
    for k in range(n_trials):
        base = 100.0 + k * 300.0
        synapse.on_pre_spike(base, None)
        for t in range(int(base) + 1, int(base + latency) + 1):
            synapse.resolve_timeouts(float(t))     # (d2)
        synapse.on_post_spike(base + latency, base)  # (e)
        synapse.resolve_timeouts(base + latency + 50.0)


def test_latency_exactly_on_the_horizon_is_causal() -> None:
    """A post-spike at exactly t_pre + verify_window is a HIT.

    BEFORE THE FIX this exact case scored hits=0, misses=50, causal_success=0.0 —
    a PERFECTLY CAUSAL synapse, whose post cell followed it every single time, was
    classified as pure noise and decayed away.

    The cause was not a debatable convention but a self-contradiction: on_post_spike
    scores a hit with `t_pre < t_post <= t_pre + verify_window` (the `<=` puts the
    boundary in the hit), while resolve_timeouts scored a miss with
    `now >= t_pre + verify_window` and ran FIRST in the tick — consuming the pending
    spike before on_post_spike could ever see it, making that `<=` branch dead code.
    resolve_timeouts now uses a strict `>`: a pre-spike is a miss only once its
    window has been PASSED, never when it is merely REACHED.
    """
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    horizon = synapse.verify_window
    assert horizon == 10.0

    _drive_at_latency(synapse, latency=horizon, n_trials=50)

    assert synapse.hits == 50
    assert synapse.misses == 0
    assert synapse.causal_success > 0.9


def test_latency_just_beyond_the_horizon_is_not_causal() -> None:
    """The fix moves the boundary by one instant; it does NOT remove the cutoff."""
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    horizon = synapse.verify_window

    _drive_at_latency(synapse, latency=horizon + 1.0, n_trials=50)

    assert synapse.hits == 0
    assert synapse.misses == 50
    assert synapse.causal_success == 0.0


# ---------------------------------------------------------------------------
# B3. The second constraint: FAN_IN * hop <= verify_window
# ---------------------------------------------------------------------------
def test_fan_in_depth_must_fit_inside_the_horizon() -> None:
    """A COROLLARY of the Law, not an exception to it.

    The DEEPEST synapse in a fan-in has latency ``fan_in * hop``. If that exceeds
    the causal horizon it scores causal_success = 0, is decayed away as noise, the
    fan-in collapses from 3 to 2, the summed input (2 x 11 = 22 mV) falls below the
    25 mV firing gap, and the assembly DIES.

    So assembly design is DOUBLY constrained, and the two pull in opposite
    directions: the Law wants ``hop`` LARGE (to keep firing sparse), the horizon
    wants it SMALL (to keep the deepest synapse causal). With fan_in = 3 and a
    10 ms horizon, hop is boxed into ~2-3 ms.
    """
    horizon = Synapse(pre_id=0, post_id=1, weight=1.0, distance=1.0).verify_window
    assert horizon == 10.0

    # hop = 3 -> deepest latency 9 ms <= 10 ms: every depth is causal.
    ok_net, ok_spikes = _ring(hop=3, run_ms=30_000)
    assert FAN_IN * 3 <= horizon
    for k in range(1, FAN_IN + 1):
        depth = [s for i in range(RING) for s in ok_net.incoming[i]
                 if s.pre_id == (i - k) % RING]
        assert all(s.causal_success > 0.5 for s in depth), f"depth k={k} not causal"
    assert _cell_rate(ok_spikes, 0, 25_000, 30_000) > 10.0  # reverberating

    # hop = 4 -> deepest latency 12 ms > 10 ms: the DEEPEST synapse is pruned.
    bad_net, bad_spikes = _ring(hop=4, run_ms=30_000)
    assert FAN_IN * 4 > horizon

    deepest = [s for i in range(RING) for s in bad_net.incoming[i]
               if s.pre_id == (i - FAN_IN) % RING]
    shallower = [s for i in range(RING) for s in bad_net.incoming[i]
                 if s.pre_id == (i - 1) % RING]

    # It fired, its post followed — but too late to be credited. Judged non-causal.
    assert all(s.causal_success == 0.0 for s in deepest)
    # ...and therefore decayed well below its (causal, protected) neighbours.
    assert statistics.mean(s.weight for s in deepest) < statistics.mean(
        s.weight for s in shallower
    )

    # The fan-in has collapsed 3 -> 2, so the assembly can no longer sustain itself.
    assert _cell_rate(bad_spikes, 0, 25_000, 30_000) == 0.0


# ===========================================================================
# [CD-3] The convergent-ring convention: simultaneity, ignition, and the
# ignition PROCEDURE. These pin the property the whole assembly rests on.
# ===========================================================================

_PROBE = Cell(neuron_id=0)
GAP = _PROBE.Vthresh - _PROBE.Vrest  # 25.0 mV, derived from the Cell itself


def _tunable_ring(fan_in: int, weight: float, hop: float, *, run_ms: int = 30_000,
                  sequential: bool = True) -> tuple[Network, list]:
    """The canonical CONVERGENT ring, with fan_in / weight / hop free.

    Synapse (i-k) -> i has delay k*hop. Because cell (i-k) fires (k-1)*hop earlier,
    all F bumps arrive on the SAME tick — see Network's convergent-ring convention.

    ``sequential=False`` ignites every seed cell at t=0 instead, which destroys that
    compensation for the FIRST wave (see T3).
    """
    net = Network(dt=1.0)
    for i in range(RING):
        net.add_cell(Cell(neuron_id=i))
    for i in range(RING):
        for k in range(1, fan_in + 1):
            net.add_synapse(_syn((i - k) % RING, i, weight, k * hop))

    spikes: list = []
    for _ in range(run_ms):
        for i in range(fan_in):
            ignite_at = i * hop if sequential else 0
            if net.current_time == ignite_at:
                net.inject(i, 100.0)
        spikes.extend(net.step())
    return net, spikes


def _reverberates(spikes: list, run_ms: int) -> bool:
    return _cell_rate(spikes, 0, run_ms - 5_000, run_ms) > 1.0


# ---------------------------------------------------------------------------
# T1. The property everything rests on: fan-in bumps arrive SIMULTANEOUSLY
# ---------------------------------------------------------------------------
def test_fan_in_bumps_arrive_simultaneously() -> None:
    """The longer distance is exactly compensated by the earlier start.

    Cell (i-k) fires (k-1)*hop earlier and its synapse has delay k*hop, so every
    bump lands on tick T+hop regardless of k. This is a DELAY LINE, and it is the
    reason fan-in can clear the 25 mV gap at all: with no interval between the
    bumps, there is no inter-bump membrane leak.

    If this ever stops holding, the ignition condition silently changes from
    F*w >= gap to a leaky staggered sum, and assemblies that used to light will go
    dark for no visible reason.
    """
    fan_in, weight, hop = 3, 11.0, 3.0
    net, _ = _tunable_ring(fan_in, weight, hop, run_ms=200)

    # Pending deliveries are (arrival_time, target_id, eff_weight, source_pre_id).
    # Every target's queued fan-in deliveries must land on ONE tick.
    by_target: dict[int, set[float]] = {}
    for arrival_time, target_id, _weight, _source in net._pending:
        by_target.setdefault(target_id, set()).add(arrival_time)

    assert by_target, "expected deliveries in flight"
    for target_id, arrivals in by_target.items():
        assert len(arrivals) == 1, (
            f"cell {target_id} has STAGGERED arrivals {sorted(arrivals)} — the "
            "convergent-ring convention is broken"
        )

    # The physical consequence: the post cell's Vm jumps by ~F*w in ONE tick, not in
    # F separate leaky steps.
    #
    # Probed with a deliberately SUB-THRESHOLD weight (3 x 5 = 15 mV < the 25 mV
    # gap): a cell that actually fires is reset to Vreset by the same integrate()
    # call, which would hide the very jump we are trying to observe. Cells 0/1/2 are
    # injected sequentially, so cell 3 sees the genuine convergent fan-in.
    probe_weight = 5.0
    assert fan_in * probe_weight < 25.0  # stays sub-threshold: no reset, no hiding

    probe = Network(dt=1.0)
    for i in range(RING):
        probe.add_cell(Cell(neuron_id=i))
    for i in range(RING):
        for k in range(1, fan_in + 1):
            probe.add_synapse(_syn((i - k) % RING, i, probe_weight, k * hop))

    jumps = []
    previous = probe.cells[3].Vm
    for _ in range(40):
        for i in range(fan_in):
            if probe.current_time == i * hop:
                probe.inject(i, 100.0)   # sequential ignition
        probe.step()
        current = probe.cells[3].Vm
        jumps.append(current - previous)
        previous = current

    # All three bumps land together: ONE tick carries the whole F*w rise...
    assert max(jumps) == pytest.approx(fan_in * probe_weight, abs=0.5)  # ~15 mV
    # ...and there is exactly one such arrival tick, not F staggered ones.
    assert sum(1 for jump in jumps if jump > 1.0) == 1


# ---------------------------------------------------------------------------
# T2. Ignition is F * w — independent of hop and tau
# ---------------------------------------------------------------------------
def test_ignition_is_fan_in_times_weight() -> None:
    """assembly_ignition_voltage(F, w) >= gap predicts reverberation exactly.

    Including the two cases a staggered-leak formula got WRONG — (2,13) and (3,9)
    both reverberate despite that formula calling them dead — and a genuinely
    sub-threshold case, (2,12) = 24 mV < 25 mV, which is born dead.
    """
    assert GAP == 25.0

    cases = [
        (3, 11.0, 3.0, True),   # 33 >= 25
        (2, 13.0, 3.0, True),   # 26 >= 25  <- a staggered-leak formula called this dead
        (3, 9.0, 2.0, True),    # 27 >= 25  <- and this
        (2, 15.0, 3.0, True),   # 30 >= 25
        (2, 12.0, 3.0, False),  # 24  < 25  <- genuinely sub-threshold
    ]
    for fan_in, weight, hop, expect_alive in cases:
        voltage = assembly_ignition_voltage(fan_in, weight)
        assert voltage == fan_in * weight  # independent of hop and tau

        run_ms = 30_000
        _net, spikes = _tunable_ring(fan_in, weight, hop, run_ms=run_ms)

        assert (voltage >= GAP) == expect_alive, f"prediction wrong: F={fan_in} w={weight}"
        assert _reverberates(spikes, run_ms) == expect_alive, (
            f"F={fan_in} w={weight} hop={hop} did not match prediction"
        )

    # The sub-threshold ring is BORN DEAD: only its injected spikes, then silence.
    # Mistaking that for a pruning failure is the likeliest misdiagnosis when scaling.
    _net, spikes = _tunable_ring(2, 12.0, 3.0, run_ms=30_000)
    assert len(spikes) == 2


# ---------------------------------------------------------------------------
# T3. The ignition PROCEDURE matters (the trap that produced false numbers)
# ---------------------------------------------------------------------------
def test_sequential_ignition_is_required() -> None:
    """Igniting all seed cells at t=0 can leave a perfectly viable ring dark forever.

    SEQUENTIAL ignition (inject(i) at t = i*hop) builds the travelling wave directly,
    so every bump is compensated and ignition is F*w.

    ALL-AT-ONCE ignition fires the seeds together, so the FIRST wave has no head
    start: it genuinely arrives STAGGERED and leaks, and that transient obeys
    V_stag = sum(w * exp(-(F-k)*hop/tau)). If V_stag < gap the ring never lights AT
    ALL — even though F*w >= gap and the travelling wave would sustain happily.

    (2, 13, 3): F*w = 26 >= 25, but V_stag = 24.19 < 25.
        sequential  -> 33.2 Hz
        all-at-once ->  0.0 Hz

    This is exactly why a staggered-leak formula once looked correct: measured under
    all-at-once ignition it WAS correct — about the transient, not the steady state.
    """
    fan_in, weight, hop, run_ms = 2, 13.0, 3.0, 30_000

    # It can SUSTAIN: F*w clears the gap.
    assert assembly_ignition_voltage(fan_in, weight) >= GAP

    # ...but the all-at-once transient does NOT clear it.
    v_stag = sum(
        weight * math.exp(-(fan_in - k) * hop / 20.0) for k in range(1, fan_in + 1)
    )
    assert v_stag < GAP
    assert v_stag == pytest.approx(24.19, abs=0.01)

    _net, sequential_spikes = _tunable_ring(
        fan_in, weight, hop, run_ms=run_ms, sequential=True
    )
    _net, all_at_once_spikes = _tunable_ring(
        fan_in, weight, hop, run_ms=run_ms, sequential=False
    )

    assert _reverberates(sequential_spikes, run_ms)        # lights and sustains
    assert not _reverberates(all_at_once_spikes, run_ms)   # never lights at all
    assert _cell_rate(all_at_once_spikes, 0, 25_000, 30_000) == 0.0


# ---------------------------------------------------------------------------
# T4. The horizon constraint is fatal when the SURVIVORS are sub-threshold
# ---------------------------------------------------------------------------
def test_horizon_constraint_is_fatal_when_survivors_are_subthreshold() -> None:
    """(F=2, w=20, hop=8): deepest latency 16 ms > the 10 ms horizon.

    The deepest synapse is judged non-causal (cs = 0.0) and decayed away, collapsing
    the fan-in 2 -> 1. The lone survivor delivers 20 mV < the 25 mV gap, so the ring
    DIES.

    Constraint 2 is therefore fatal exactly when (F - 1) * w < gap. It is a
    MECHANISM, not an absolute prohibition: an assembly whose survivors still clear
    the gap would limp on after losing its deepest synapse. (An earlier claim that
    this specific configuration survives at 6 Hz was wrong — it dies.)
    """
    fan_in, weight, hop, run_ms = 2, 20.0, 8.0, 30_000
    net, spikes = _tunable_ring(fan_in, weight, hop, run_ms=run_ms)

    horizon = net.incoming[0][0].verify_window
    assert fan_in * hop > horizon  # 16 > 10: the deepest synapse is out of sight

    # It COULD have ignited — this is NOT a sub-threshold ring.
    assert assembly_ignition_voltage(fan_in, weight) >= GAP  # 40 >= 25

    deepest = [s for i in range(RING) for s in net.incoming[i]
               if s.pre_id == (i - fan_in) % RING]
    shallower = [s for i in range(RING) for s in net.incoming[i]
                 if s.pre_id == (i - 1) % RING]

    # The deepest synapse is judged pure noise and decayed away.
    assert all(s.causal_success == 0.0 for s in deepest)
    assert statistics.mean(s.weight for s in deepest) < statistics.mean(
        s.weight for s in shallower
    )

    # Fan-in collapses 2 -> 1, and one 20 mV bump cannot clear the 25 mV gap.
    assert (fan_in - 1) * weight < GAP
    assert not _reverberates(spikes, run_ms)
    assert _cell_rate(spikes, 0, 25_000, 30_000) == 0.0
