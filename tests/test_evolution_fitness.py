from misterdev.core.evolution import FitnessScore, estimate_noise_band


def test_rates_and_per_task_costs():
    s = FitnessScore(resolved=30, total=120, cost=6.0)
    assert s.resolved_rate == 0.25
    assert s.cost_per_task == 0.05


def test_empty_suite_is_zero_not_crash():
    s = FitnessScore(resolved=0, total=0, cost=0.0)
    assert s.resolved_rate == 0.0
    assert s.cost_per_task == 0.0


def test_from_report_reads_duck_typed_report():
    class _Report:
        resolved = 7
        total = 10

    s = FitnessScore.from_report(_Report(), cost=2.5, regressions=1)
    assert (s.resolved, s.total, s.cost, s.regressions) == (7, 10, 2.5, 1)


def test_gain_beyond_band_is_a_win():
    base = FitnessScore(50, 100, 1.0)
    cand = FitnessScore(60, 100, 1.0)  # +10 points
    assert cand.beats(base, noise_band=0.05)


def test_gain_inside_band_is_not_a_win():
    base = FitnessScore(50, 100, 1.0)
    cand = FitnessScore(53, 100, 1.0)  # +3 points, inside a 5% band
    assert not cand.beats(base, noise_band=0.05)


def test_any_regression_disqualifies_even_a_large_gain():
    base = FitnessScore(50, 100, 1.0)
    cand = FitnessScore(90, 100, 1.0, regressions=1)
    assert not cand.beats(base, noise_band=0.05)


def test_quality_tie_breaks_on_cost():
    base = FitnessScore(50, 100, 2.0)
    cheaper = FitnessScore(50, 100, 1.0)  # same rate, half the cost
    pricier = FitnessScore(50, 100, 3.0)
    assert cheaper.beats(base, noise_band=0.05)
    assert not pricier.beats(base, noise_band=0.05)


def test_quality_regression_within_band_is_not_rescued_by_cost():
    # A rate DROP inside the band still must not be adopted just for being cheaper
    # only when the drop is within band AND cost is worse; a clear drop loses.
    base = FitnessScore(50, 100, 2.0)
    worse_rate = FitnessScore(44, 100, 0.1)  # -6 points exceeds a 5% band -> loses
    assert not worse_rate.beats(base, noise_band=0.05)


def test_estimate_noise_band_is_stdev():
    rates = [0.50, 0.50, 0.50]
    assert estimate_noise_band(rates) == 0.0  # no spread -> no noise
    band = estimate_noise_band([0.40, 0.60])
    assert abs(band - 0.10) < 1e-9  # population stdev of {.4,.6}


def test_estimate_noise_band_needs_two_samples():
    assert estimate_noise_band([]) == 0.0
    assert estimate_noise_band([0.5]) == 0.0
