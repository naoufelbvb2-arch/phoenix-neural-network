# Phoenix — a falsification study of a temporal spiking neural network

Phoenix is a spiking neural network (delay lines, coincidence detection, STDP/CST plasticity,
E/I balance). This repository is not a demo of it working — it is a **falsification program**: a
series of pre-registered experiments, each with a mandatory matched control, asking whether the
network's apparent "self-organisation" is real or an artifact of how it was measured.

**Headline result.** Six independent emergence claims were tested. **All six dissolved under a
matched control** — and five of them produced a promising number *before* the control was added.
What survives is small, designed, and temporal: a **feedforward delay-line + coincidence detector**,
configured by two distributional facts (log-normal weights, and a delay *mean* timed to the readout).
Recurrence, plasticity, and network scale each turned out to add nothing their controls didn't.

> The guiding rule of the whole effort: **a promising number is a hypothesis about a confound until
> a matched control rules it out** — including the project's own positive results.

The full write-up is [`WRITEUP_controls_as_method.md`](WRITEUP_controls_as_method.md) (rendered as
[`paper.html`](paper.html)).

---

## The six dissolutions

| # | Claim | Control that dissolved it |
|---|-------|---------------------------|
| 1 | Recurrent computation | rate/structure-matched feed-forward → it was delay lines |
| 2 | STDP learns a useful gain | random-input arm scored identically → the 92% was the w_max clamp |
| 3 | Synaptic scaling shapes weights | the spread is better drawn a priori (log-normal) |
| 4 | State space encodes order | delete recurrent synapses → separation unchanged |
| 5 | Capacity scales with N | connected 64k ≡ four disconnected 16k blocks (partition test) |
| 6 | Delay plasticity learns cue structure | cue-trained ≈ random-trained (two rules, 5 seeds) |

Plus **Task 4 (spatial geometry)**: distance-derived delays ≈ shuffled → spatial correlation adds
nothing; geometry is just a (worse) delay-marginal generator.

**Constructive test.** The architecture the analysis implies — a pure feedforward delay-line +
coincidence detector, *no cycles, no inhibition, no plasticity/homeostasis* — **matches** the recurrent
network on capacity (fair readout), equals-or-beats it on memory depth, is more readout-robust, and
discards the entire stability apparatus (no runaway, no ignition, no homeostatic controller, provable
termination). An early "6× beats" number was **withdrawn** — it was a readout-selection artifact.

---

## Repository layout

**Compute core**
- `phoenix/` — the model. `cell.py`, `synapse.py`, `network_graph.py` are the OOP **oracle** (the
  mathematical reference). `soa.py` is the vectorized **Struct-of-Arrays** layer, bit-exact against the
  oracle and ~50× faster (bucket-ring delivery via `np.add.at`).
- `phoenix_harness.py` — the self-contained experimental harness (build, fingerprints, decoder,
  capacity/memory tasks). Run `python phoenix_harness.py` to validate it.
- `tests/` — 225+ tests; the SoA layer is gated bit-for-bit against the OOP oracle.

**Experiments & findings** (`experiments/`)
- `PREREG_*.md` — pre-registrations, locked (criterion + predictions) *before* running.
- `FINDING_*.md` — results: `capacity_does_not_scale`, `task3_delay_plasticity`, `delay_span_probe`,
  `constructive_test`.
- `task*.py`, `delay_span_probe.py`, `constructive_test.py` — the experiment scripts; `*.json` — their
  committed data (seeds fixed, so results regenerate exactly).

**Write-up & reproduction**
- `WRITEUP_controls_as_method.md` / `paper.html` — the methods paper.
- `reproduce.py` — clean-room reproduction: regenerates every headline number from committed seeds,
  prints measured vs published, exits non-zero on any mismatch.

## Reproduce it

```bash
python phoenix_harness.py          # validate the harness reproduces its documented output
python -m pytest -q                # 225+ tests, SoA bit-exact vs the OOP oracle
python reproduce.py                # regenerate headline numbers from a clean state (~1h, N<=2000)
python reproduce.py --full         # + the 16k/64k capacity/partition checks (hours)
```

Last clean-room run: **6/6 headline numbers reproduced within tolerance from a fresh clone, in 27
minutes** (log in `experiments/reproduce_log.txt`). The SoA layer is deterministic and bit-exact, so
the same seeds give the same numbers.

---

## Project status — STOPPED 2026-07-28 (complete)

The falsification program is **finished**. All brief items are closed and the record is committed:

- **Tasks 0–4:** done. State-space order refuted; capacity scaling closed (negative); sparsity/cost
  obstructed; delay plasticity is entry six (no cue-specific structure); geometry closed (spatial
  structure irrelevant).
- **Delay-span lever:** settled — feedforward propagation depth, not held recurrent memory.
- **Constructive test:** the simpler feedforward architecture reproduces the recurrent one; the
  stability apparatus is unnecessary. (The 6× capacity "win" was withdrawn as a readout artifact.)
- **Reproduction:** 6/6, clean-room, passed.
- **Write-up:** complete and published as a private artifact (share is the author's choice).

**Open threads deliberately not pursued** (each would be a fresh pre-registration, not a continuation):
- The delay-span **optimum and its N-scaling** (once lag is held at a fixed lag/span ratio) — the one
  remaining probe of the surviving positive; predicted to flatten under an absolute-cue control.
- A **latency-coded / sparse-distributed encoder** (the brief's §8 forward context) — never built.
- The **256k capacity point** — unnecessary once the partition test closed Question A.

The scope is honest and stated in the paper: one codebase, synthetic tasks, no real data, single
implementations. The *method* generalises; the *results* are about this architecture on these tasks.
