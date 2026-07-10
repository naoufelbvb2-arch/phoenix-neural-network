"""Tests for the Phoenix ``Synapse`` connection object."""

import math

from phoenix.spike import Spike
from phoenix.synapse import Synapse


def test_delay_proportional_to_distance() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=1.0, distance=10.0, propagation_speed=2.0
    )
    assert synapse.delay == 5.0


def test_delay_zero_distance() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=1.0, distance=0.0, propagation_speed=2.0
    )
    assert synapse.delay == 0.0


def test_effective_weight_equals_base_weight_at_zero_distance() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=7.5, distance=0.0)
    assert synapse.effective_weight() == 7.5


def test_effective_weight_decreases_with_distance() -> None:
    near = Synapse(pre_id=1, post_id=2, weight=10.0, distance=5.0, decay_constant=10.0)
    far = Synapse(pre_id=1, post_id=2, weight=10.0, distance=50.0, decay_constant=10.0)

    assert far.effective_weight() < near.effective_weight()


def test_effective_weight_exact_formula() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=10.0, distance=20.0, decay_constant=10.0
    )
    expected = 10.0 * math.exp(-20.0 / 10.0)
    assert abs(synapse.effective_weight() - expected) < 1e-9


def test_propagate_computes_correct_arrival_time() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=1.0, distance=15.0, propagation_speed=3.0
    )
    spike = Spike(neuron_id=1, timestamp=100.0)

    arrival_time, _ = synapse.propagate(spike)

    assert arrival_time == 105.0


def test_propagate_returns_effective_weight_not_base_weight() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=30.0, decay_constant=10.0
    )
    spike = Spike(neuron_id=1, timestamp=0.0)

    _, delivered_weight = synapse.propagate(spike)

    assert delivered_weight == synapse.effective_weight()
    assert delivered_weight != synapse.weight


def test_synapse_stores_pre_and_post_ids() -> None:
    synapse = Synapse(pre_id=3, post_id=7, weight=1.0, distance=1.0)
    assert synapse.pre_id == 3
    assert synapse.post_id == 7


def test_causal_timing_potentiates_weight() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    initial_weight = synapse.weight

    synapse.update_weight(t_pre=10.0, t_post=15.0)

    assert synapse.weight > initial_weight


def test_anticausal_timing_depresses_weight() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    initial_weight = synapse.weight

    synapse.update_weight(t_pre=15.0, t_post=10.0)

    assert synapse.weight < initial_weight


def test_simultaneous_spikes_no_change() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    initial_weight = synapse.weight

    synapse.update_weight(t_pre=10.0, t_post=10.0)

    assert abs(synapse.weight - initial_weight) < 1e-9


def test_potentiation_decays_with_larger_delta_t() -> None:
    close = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    far = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    close.update_weight(t_pre=0.0, t_post=2.0)
    far.update_weight(t_pre=0.0, t_post=50.0)

    close_increase = close.weight - 5.0
    far_increase = far.weight - 5.0
    assert close_increase > far_increase


def test_depression_decays_with_larger_magnitude_delta_t() -> None:
    close = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    far = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    close.update_weight(t_pre=2.0, t_post=0.0)  # delta_t = -2.0
    far.update_weight(t_pre=50.0, t_post=0.0)  # delta_t = -50.0

    close_decrease = 5.0 - close.weight
    far_decrease = 5.0 - far.weight
    assert close_decrease > far_decrease


def test_weight_respects_upper_bound() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=19.9, distance=1.0, learning_rate=1.0, w_max=20.0
    )

    for _ in range(20):
        synapse.update_weight(t_pre=0.0, t_post=1.0)  # strongly causal
        assert synapse.weight <= synapse.w_max

    assert synapse.weight == synapse.w_max


def test_weight_respects_lower_bound() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=0.1, distance=1.0, learning_rate=1.0, w_min=0.0
    )

    for _ in range(20):
        synapse.update_weight(t_pre=1.0, t_post=0.0)  # strongly anti-causal
        assert synapse.weight >= synapse.w_min

    assert synapse.weight == synapse.w_min


def test_learning_rate_is_externally_mutable() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, learning_rate=0.5)

    synapse.update_weight(t_pre=10.0, t_post=15.0)
    weight_after_first_update = synapse.weight

    synapse.learning_rate = 0.0
    synapse.update_weight(t_pre=10.0, t_post=15.0)

    assert synapse.weight == weight_after_first_update


def test_asymmetric_A_plus_A_minus() -> None:
    baseline = Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=1.0, A_plus=1.0, A_minus=1.0
    )
    boosted = Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=1.0, A_plus=2.0, A_minus=1.0
    )

    baseline.update_weight(t_pre=0.0, t_post=5.0)
    boosted.update_weight(t_pre=0.0, t_post=5.0)

    baseline_increase = baseline.weight - 5.0
    boosted_increase = boosted.weight - 5.0
    assert boosted_increase > baseline_increase


def test_record_observation_stores_positive_delay() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    synapse.record_observation(t_pre=10.0, t_post=25.0)

    assert synapse.observed_delays == [15.0]


def test_record_observation_ignores_non_causal_pairs() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    synapse.record_observation(t_pre=25.0, t_post=10.0)  # post before pre
    synapse.record_observation(t_pre=10.0, t_post=10.0)  # simultaneous

    assert synapse.observed_delays == []


def test_observed_delays_respects_max_history() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, max_history=3)

    for delay in (10.0, 20.0, 30.0, 40.0, 50.0):
        synapse.record_observation(t_pre=0.0, t_post=delay)

    assert len(synapse.observed_delays) == 3
    assert synapse.observed_delays == [30.0, 40.0, 50.0]


def test_mean_observed_delay_none_when_empty() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert synapse.mean_observed_delay is None


def test_mean_observed_delay_correct_value() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for delay in (10.0, 20.0, 30.0):
        synapse.record_observation(t_pre=0.0, t_post=delay)

    assert synapse.mean_observed_delay == 20.0


def test_delay_variance_none_with_fewer_than_two_observations() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert synapse.delay_variance is None

    synapse.record_observation(t_pre=0.0, t_post=10.0)
    assert synapse.delay_variance is None


def test_delay_variance_correct_value() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for delay in (10.0, 20.0, 30.0):
        synapse.record_observation(t_pre=0.0, t_post=delay)

    expected = ((10.0 - 20.0) ** 2 + (20.0 - 20.0) ** 2 + (30.0 - 20.0) ** 2) / 3
    assert abs(synapse.delay_variance - expected) < 1e-6


def test_delay_variance_zero_for_perfectly_regular_delays() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(3):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    assert synapse.delay_variance == 0.0


def test_record_observation_independent_of_update_weight() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    initial_weight = synapse.weight

    synapse.record_observation(t_pre=10.0, t_post=25.0)
    synapse.record_observation(t_pre=0.0, t_post=15.0)

    assert synapse.weight == initial_weight


def test_confidence_none_with_fewer_than_two_observations() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert synapse.confidence is None

    synapse.record_observation(t_pre=0.0, t_post=10.0)
    assert synapse.confidence is None


def test_confidence_high_for_regular_delays_with_enough_samples() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(20):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    # CV=0 -> regularity=exp(0)=1.0. n=20, n0=5 -> sample_factor=20/25=0.8.
    # confidence = 1.0 * 0.8 = 0.8.
    assert synapse.confidence > 0.7


def test_confidence_low_for_irregular_delays() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for delay in (5.0, 50.0, 12.0, 80.0, 3.0, 60.0):
        synapse.record_observation(t_pre=0.0, t_post=delay)

    # mean=35.0, population variance=888.0, CV=sqrt(888)/35≈0.851,
    # regularity=exp(-0.851)≈0.427, n=6, n0=5 -> sample_factor=6/11≈0.545,
    # confidence≈0.233.
    assert synapse.confidence < 0.3


def test_confidence_increases_with_sample_size_at_same_regularity() -> None:
    few = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    many = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    for _ in range(3):
        few.record_observation(t_pre=0.0, t_post=10.0)
    for _ in range(30):
        many.record_observation(t_pre=0.0, t_post=10.0)

    assert many.confidence > few.confidence


def test_confidence_matches_exact_formula() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for delay in (10.0, 10.0, 10.0, 20.0):
        synapse.record_observation(t_pre=0.0, t_post=delay)

    delays = [10.0, 10.0, 10.0, 20.0]
    mean = sum(delays) / len(delays)
    variance = sum((d - mean) ** 2 for d in delays) / len(delays)
    cv = math.sqrt(variance) / mean
    regularity = math.exp(-cv)
    n = len(delays)
    sample_factor = n / (n + synapse.n0)
    expected = regularity * sample_factor

    assert abs(synapse.confidence - expected) < 1e-6


def test_future_expectation_none_when_no_observations() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert synapse.future_expectation is None


def test_future_expectation_usable_with_single_observation() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    synapse.record_observation(t_pre=0.0, t_post=12.0)

    assert synapse.future_expectation == 12.0
    assert synapse.confidence is None  # still undefined at n=1


def test_future_expectation_equals_mean_observed_delay() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for delay in (10.0, 20.0, 30.0):
        synapse.record_observation(t_pre=0.0, t_post=delay)

    assert synapse.future_expectation == synapse.mean_observed_delay


def test_confidence_defensive_against_zero_mean_delay() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    # record_observation can never produce a mean of exactly 0.0 (it only
    # accepts observed_delay > 0), so simulate the otherwise-unreachable
    # edge case directly to confirm the defensive guard, not a crash.
    synapse.observed_delays = [0.0, 0.0]

    assert synapse.confidence is None


def test_prediction_error_none_with_no_prior_expectation() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert synapse.compute_prediction_error(t_pre=10.0, t_post=20.0) is None


def test_prediction_error_none_for_non_causal_pair() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(3):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    assert synapse.compute_prediction_error(t_pre=20.0, t_post=10.0) is None  # post before pre
    assert synapse.compute_prediction_error(t_pre=20.0, t_post=20.0) is None  # simultaneous


def test_prediction_error_zero_when_matches_expectation_exactly() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(5):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    error = synapse.compute_prediction_error(t_pre=100.0, t_post=115.0)

    assert abs(error - 0.0) < 1e-9


def test_prediction_error_large_when_mismatched() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(5):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    error = synapse.compute_prediction_error(t_pre=100.0, t_post=160.0)  # actual delay = 60.0

    assert abs(error - 45.0) < 1e-9


def test_prediction_error_does_not_mutate_state() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(5):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    expectation_before = synapse.future_expectation
    history_length_before = len(synapse.observed_delays)
    weight_before = synapse.weight

    synapse.compute_prediction_error(t_pre=100.0, t_post=115.0)
    synapse.compute_prediction_error(t_pre=100.0, t_post=160.0)
    synapse.compute_prediction_error(t_pre=100.0, t_post=90.0)  # non-causal

    assert synapse.future_expectation == expectation_before
    assert len(synapse.observed_delays) == history_length_before
    assert synapse.weight == weight_before


def test_weighted_error_falls_back_to_raw_when_confidence_none() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    synapse.record_observation(t_pre=0.0, t_post=15.0)  # n=1: expectation exists, confidence doesn't
    assert synapse.confidence is None

    raw_error = synapse.compute_prediction_error(t_pre=100.0, t_post=160.0)
    weighted_error = synapse.compute_weighted_prediction_error(t_pre=100.0, t_post=160.0)

    assert weighted_error == raw_error


def test_weighted_error_scales_down_with_low_confidence() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for delay in (5.0, 50.0, 12.0, 80.0, 3.0, 60.0):  # irregular -> low confidence
        synapse.record_observation(t_pre=0.0, t_post=delay)
    assert synapse.confidence is not None
    assert synapse.confidence < 1.0

    raw_error = synapse.compute_prediction_error(t_pre=100.0, t_post=160.0)
    weighted_error = synapse.compute_weighted_prediction_error(t_pre=100.0, t_post=160.0)

    assert abs(weighted_error) < abs(raw_error)


def test_weighted_error_close_to_raw_with_high_confidence() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(50):  # highly regular, many samples -> confidence near (but below) 1.0
        synapse.record_observation(t_pre=0.0, t_post=15.0)
    assert synapse.confidence > 0.9

    raw_error = synapse.compute_prediction_error(t_pre=100.0, t_post=160.0)
    weighted_error = synapse.compute_weighted_prediction_error(t_pre=100.0, t_post=160.0)

    assert weighted_error < raw_error
    assert abs(weighted_error - raw_error) < raw_error * 0.15


def test_call_order_matters_conceptually() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(5):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    # Correct usage: compute the error against the PRIOR expectation first...
    error = synapse.compute_prediction_error(t_pre=100.0, t_post=115.0)
    assert abs(error - 0.0) < 1e-9

    # ...then fold the new observation in. Since compute_prediction_error
    # never mutates state, calling it doesn't itself change the baseline —
    # the call-order guidance is about which value the CALLER treats as
    # "the" prediction error for this event, not a safety hazard in the code.
    synapse.record_observation(t_pre=100.0, t_post=115.0)
    assert synapse.future_expectation == 15.0


def test_modulation_neutral_when_no_expectation_exists() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    assert synapse.compute_modulation(t_pre=10.0, t_post=25.0) == 1.0


def test_modulation_near_m_min_for_perfectly_predicted_event() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(50):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    # actual delay (100 -> 115) matches future_expectation exactly ->
    # weighted_error == 0.0 -> modulation collapses exactly to m_min.
    modulation = synapse.compute_modulation(t_pre=100.0, t_post=115.0)

    assert abs(modulation - synapse.m_min) < 1e-9


def test_modulation_approaches_m_max_for_highly_surprising_event() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    for _ in range(50):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    # actual delay is 500.0, wildly off the established ~15.0 pattern.
    modulation = synapse.compute_modulation(t_pre=100.0, t_post=600.0)

    assert abs(modulation - synapse.m_max) < 0.01


def test_modulation_respects_custom_bounds() -> None:
    synapse = Synapse(
        pre_id=1, post_id=2, weight=5.0, distance=1.0, m_min=0.0, m_max=5.0
    )
    for _ in range(50):
        synapse.record_observation(t_pre=0.0, t_post=15.0)

    matching_modulation = synapse.compute_modulation(t_pre=100.0, t_post=115.0)
    mismatched_modulation = synapse.compute_modulation(t_pre=100.0, t_post=600.0)

    assert 0.0 <= matching_modulation <= 5.0
    assert 0.0 <= mismatched_modulation <= 5.0
    assert abs(matching_modulation - 0.0) < 1e-9
    assert abs(mismatched_modulation - 5.0) < 0.01
    assert mismatched_modulation > matching_modulation


def test_update_weight_uses_modulated_rate_not_raw_rate() -> None:
    low_surprise = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    high_surprise = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    # low_surprise expects ~15.0 (matches the event applied below).
    for _ in range(20):
        low_surprise.record_observation(t_pre=0.0, t_post=15.0)
    # high_surprise expects ~200.0 (wildly mismatches that same event).
    for _ in range(20):
        high_surprise.record_observation(t_pre=0.0, t_post=200.0)

    # Identical (t_pre, t_post) -> identical delta_t=15.0 for the STDP part
    # itself; only each synapse's own established expectation differs.
    low_surprise.update_weight(t_pre=1000.0, t_post=1015.0)
    high_surprise.update_weight(t_pre=1000.0, t_post=1015.0)

    low_change = abs(low_surprise.weight - 5.0)
    high_change = abs(high_surprise.weight - 5.0)
    assert high_change > low_change


def test_update_weight_unmodulated_for_fresh_synapse() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    synapse.update_weight(t_pre=10.0, t_post=15.0)

    expected_dw = 0.01 * 1.0 * math.exp(-5.0 / 20.0)  # learning_rate * A_plus * exp(...)
    assert abs(synapse.weight - (5.0 + expected_dw)) < 1e-9


def test_trace_decays_exponentially_between_events() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, tau_stdp=20.0)
    synapse.pre_trace = 1.0
    synapse.last_trace_update = 0.0

    synapse._decay_traces(current_time=10.0)

    expected = 1.0 * math.exp(-10.0 / 20.0)
    assert abs(synapse.pre_trace - expected) < 1e-9
    assert synapse.last_trace_update == 10.0


def test_on_pre_spike_increments_pre_trace() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    synapse.on_pre_spike(t_pre=10.0, t_post_partner=None)

    assert synapse.pre_trace == 1.0


def test_on_post_spike_uses_pre_trace_for_potentiation() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    synapse.on_pre_spike(t_pre=10.0, t_post_partner=None)
    synapse.on_post_spike(t_post=15.0, t_pre_partner=10.0)

    assert synapse.weight > 5.0


def test_trace_equivalence_to_multi_pair_sum_analytically() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0, tau_stdp=20.0)

    for t_pre in (10.0, 12.0, 14.0):
        synapse.on_pre_spike(t_pre=t_pre, t_post_partner=None)

    # Closed-form sum over every past pre-spike (including the one AT
    # t=14.0 itself, contributing exp(0)=1 after its own increment) —
    # proving trace accumulation is exactly equivalent to an all-pairs sum
    # under a pure exponential kernel, not merely "some decaying number."
    expected_pre_trace = sum(
        math.exp(-(14.0 - t_i) / 20.0) for t_i in (10.0, 12.0, 14.0)
    )
    assert abs(synapse.pre_trace - expected_pre_trace) < 1e-9


def test_on_pre_spike_forward_direction_does_not_record_observation() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    # t_post_partner (10.0) is EARLIER than t_pre (20.0) -> anti-causal from
    # record_observation's perspective; on_pre_spike's own guard only calls
    # record_observation when that ordering holds, and record_observation
    # itself then rejects it as non-causal -> always a documented no-op.
    synapse.on_pre_spike(t_pre=20.0, t_post_partner=10.0)

    assert synapse.observed_delays == []


def test_on_post_spike_forward_direction_records_observation() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)

    synapse.on_post_spike(t_post=25.0, t_pre_partner=10.0)

    assert synapse.observed_delays == [15.0]


def test_update_weight_still_works_independently() -> None:
    synapse = Synapse(pre_id=1, post_id=2, weight=5.0, distance=1.0)
    initial_weight = synapse.weight

    synapse.update_weight(t_pre=10.0, t_post=15.0)

    assert synapse.weight > initial_weight
