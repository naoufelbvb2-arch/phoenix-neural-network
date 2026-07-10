"""Spike events emitted by a Phoenix ``Cell``.

A spike is a first-class, identifiable event rather than a bare boolean or
float — its origin and exact timing matter for anything built on top of it
(synchrony detection, STDP, frequency analysis).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Spike:
    """A single spike event fired by a neuron.

    Attributes:
        neuron_id: Identifier of the cell that fired.
        timestamp: Value of the firing cell's clock (``t``) at emission.
        amplitude: Fixed spike amplitude in mV (constant for now).
    """

    neuron_id: int
    timestamp: float
    amplitude: float = 40.0
