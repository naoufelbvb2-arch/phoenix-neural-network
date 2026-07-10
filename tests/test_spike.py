"""Tests for the Phoenix ``Spike`` event object."""

from phoenix.spike import Spike


def test_spike_default_amplitude() -> None:
    spike = Spike(neuron_id=1, timestamp=12.5)
    assert spike.amplitude == 40.0


def test_spike_holds_identity_and_timing() -> None:
    spike = Spike(neuron_id=42, timestamp=7.0, amplitude=40.0)
    assert spike.neuron_id == 42
    assert spike.timestamp == 7.0
    assert spike.amplitude == 40.0


def test_spike_is_never_falsy() -> None:
    spike = Spike(neuron_id=1, timestamp=0.0, amplitude=0.0)
    assert spike is not None
    assert bool(spike) is True
