"""[SOA build] O(1) duplicate check, the bulk synapse API, and the vectorized build.

The build was O(N^2): add_cell scanned a growing list for duplicates (measured
149 s of a 157 s build at N=1e5). Fixed with a set. add_synapses_bulk adds the
array API; the singular API is unchanged and BUFFERED so a mixed singular/bulk
build is byte-identical to a pure-singular one (the stable sort downstream, and
thus spike delivery order, depends on that).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from phoenix.soa import SoANetwork


def _arrays(net: SoANetwork) -> tuple[bytes, bytes, bytes, bytes]:
    return (
        net._syn_post_id.tobytes(),
        net._syn_eff_w.tobytes(),
        net._syn_delay.tobytes(),
        net._syn_indptr.tobytes(),
    )


def _spike_train(net: SoANetwork, ticks: int, seed: int) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    train: list[tuple[int, float]] = []
    for _ in range(ticks):
        for _ in range(3):
            net.inject(int(rng.integers(len(net._ids))), 100.0)
        t_after = net.cells.t + net.dt
        for nid in net.step():
            train.append((nid, t_after))
    return train


def _random_edges(n: int, e: int, seed: int) -> list[tuple[int, int, float, float]]:
    rng = np.random.default_rng(seed)
    return [
        (int(rng.integers(n)), int(rng.integers(n)),
         float(rng.uniform(2.0, 10.0)), float(rng.integers(1, 8)))
        for _ in range(e)
    ]


# ---------------------------------------------------------------------------
# 1. Bulk is byte-identical to singular (arrays AND full spike train)
# ---------------------------------------------------------------------------
def test_bulk_byte_identical_to_singular() -> None:
    n, edges = 300, _random_edges(300, 4000, seed=1)

    singular = SoANetwork(dt=1.0)
    for c in range(n):
        singular.add_cell(c)
    for p, q, w, d in edges:
        singular.add_synapse(p, q, w, d, decay_constant=20.0)
    singular._build()

    bulk = SoANetwork(dt=1.0)
    for c in range(n):
        bulk.add_cell(c)
    bulk.add_synapses_bulk(
        pre_ids=[e[0] for e in edges], post_ids=[e[1] for e in edges],
        weights=[e[2] for e in edges], distances=[e[3] for e in edges],
        decay_constant=20.0,
    )
    bulk._build()

    assert _arrays(bulk) == _arrays(singular)
    assert _spike_train(bulk, 150, seed=7) == _spike_train(singular, 150, seed=7)


# ---------------------------------------------------------------------------
# 2. Mixed insertion order (singular -> bulk -> singular) == pure singular
# ---------------------------------------------------------------------------
def test_mixed_order_identical_to_pure_singular() -> None:
    n, edges = 200, _random_edges(200, 3000, seed=2)
    a, b = 1000, 2200  # split points

    pure = SoANetwork(dt=1.0)
    for c in range(n):
        pure.add_cell(c)
    for p, q, w, d in edges:
        pure.add_synapse(p, q, w, d, decay_constant=20.0)
    pure._build()

    mixed = SoANetwork(dt=1.0)
    for c in range(n):
        mixed.add_cell(c)
    for p, q, w, d in edges[:a]:                       # singular
        mixed.add_synapse(p, q, w, d, decay_constant=20.0)
    mixed.add_synapses_bulk(                           # bulk
        pre_ids=[e[0] for e in edges[a:b]], post_ids=[e[1] for e in edges[a:b]],
        weights=[e[2] for e in edges[a:b]], distances=[e[3] for e in edges[a:b]],
        decay_constant=20.0,
    )
    for p, q, w, d in edges[b:]:                       # singular again
        mixed.add_synapse(p, q, w, d, decay_constant=20.0)
    mixed._build()

    assert _arrays(mixed) == _arrays(pure)
    assert _spike_train(mixed, 150, seed=9) == _spike_train(pure, 150, seed=9)


# ---------------------------------------------------------------------------
# 3. Scalar broadcasting in bulk
# ---------------------------------------------------------------------------
def test_bulk_scalar_broadcasting() -> None:
    n, e = 100, 500
    rng = np.random.default_rng(3)
    pre = rng.integers(0, n, e)
    post = rng.integers(0, n, e)

    scalar = SoANetwork(dt=1.0)
    for c in range(n):
        scalar.add_cell(c)
    scalar.add_synapses_bulk(pre, post, weights=8.0, distances=3.0,
                             propagation_speed=1.0, decay_constant=20.0)
    scalar._build()

    explicit = SoANetwork(dt=1.0)
    for c in range(n):
        explicit.add_cell(c)
    explicit.add_synapses_bulk(
        pre, post, weights=np.full(e, 8.0), distances=np.full(e, 3.0),
        propagation_speed=np.full(e, 1.0), decay_constant=np.full(e, 20.0),
    )
    explicit._build()

    assert _arrays(scalar) == _arrays(explicit)


# ---------------------------------------------------------------------------
# 4. Bulk validates IDs and lengths, with the verbatim messages
# ---------------------------------------------------------------------------
def test_bulk_rejects_bad_input() -> None:
    net = SoANetwork(dt=1.0)
    for c in range(10):
        net.add_cell(c)

    # Unregistered pre_id -> reported at build, verbatim, first offender in order.
    bad_pre = SoANetwork(dt=1.0)
    for c in range(10):
        bad_pre.add_cell(c)
    bad_pre.add_synapses_bulk([0, 99], [1, 2], 5.0, 2.0)
    with pytest.raises(ValueError, match="synapse pre_id 99 is not a registered cell"):
        bad_pre._build()

    bad_post = SoANetwork(dt=1.0)
    for c in range(10):
        bad_post.add_cell(c)
    bad_post.add_synapses_bulk([0, 1], [1, 77], 5.0, 2.0)
    with pytest.raises(ValueError, match="synapse post_id 77 is not a registered cell"):
        bad_post._build()

    # Mismatched pre/post lengths -> "same length".
    with pytest.raises(ValueError, match="same length"):
        net.add_synapses_bulk([0, 1, 2], [1, 2], 5.0, 2.0)

    # weights not scalar and wrong length -> "scalar or length".
    with pytest.raises(ValueError, match="scalar or length"):
        net.add_synapses_bulk([0, 1], [1, 2], [5.0, 6.0, 7.0], 2.0)


# ---------------------------------------------------------------------------
# 5. Duplicate neuron_id still raises (now via the O(1) set)
# ---------------------------------------------------------------------------
def test_duplicate_neuron_id_still_raises() -> None:
    net = SoANetwork(dt=1.0)
    net.add_cell(5)
    with pytest.raises(ValueError, match="duplicate neuron_id 5"):
        net.add_cell(5)


# ---------------------------------------------------------------------------
# 6. Quadratic-regression guard: 4x the cells must be well under 8x the time
# ---------------------------------------------------------------------------
def test_build_is_not_quadratic() -> None:
    """O(N^2) would make 16,000 cells cost ~16x the 4,000-cell build; linear ~4x.

    The threshold is a generous 8x so this is not flaky while still failing hard
    if the O(N) duplicate scan (or any other quadratic step) comes back.
    """
    def build_time(n: int) -> float:
        rng = np.random.default_rng(n)
        pre = rng.integers(0, n, 4 * n)
        post = rng.integers(0, n, 4 * n)
        start = time.perf_counter()
        net = SoANetwork(dt=1.0)
        for c in range(n):
            net.add_cell(c)
        net.add_synapses_bulk(pre, post, 8.0, 2.0)
        net._build()
        return time.perf_counter() - start

    build_time(4_000)  # warm up (imports, caches)
    small = build_time(4_000)
    large = build_time(16_000)

    assert large < 8.0 * small, f"4x cells took {large / small:.1f}x time (quadratic?)"
