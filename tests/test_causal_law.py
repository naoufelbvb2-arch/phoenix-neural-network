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
from phoenix.network_graph import Network
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
def test_two_cell_loop_is_killed_by_a_single_spike() -> None:
    """One stray spike ends a 2-cell loop permanently. Mechanism:

    the intruder makes the post cell fire EARLY; that cell enters refractory
    having already spent its spike; the real loop bump then arrives DURING
    refractory and is hard-rejected (Option A); the chain is broken, and nothing
    re-ignites it.

    MEASURED CORRECTION to the original claim: the kill is PHASE-DEPENDENT, not
    universal. Scanning the loop's 4 ms cycle, a single spike kills it in exactly
    5 of 20 phases (25%) — precisely those landing in the vulnerable quarter. It
    is pinned here at such a phase (t=100). The fragility is nonetheless real:
    with any SUSTAINED noise, a strike in the vulnerable quarter — and therefore
    permanent death — is only a matter of time. That is why assemblies (L4), not
    2-cell loops, are the unit of reverberation.
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
