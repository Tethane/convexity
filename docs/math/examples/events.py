"""Supplied surprise laws, scheduled catalysts and marked random event arrivals."""

import numpy as np
from convexity.math import distributions as d, regimes as r


def main():
    rng = np.random.default_rng(121)
    release = r.ScheduledEvent(
        2.0,
        d.student_t(5, scale=0.4),
        lambda surprise: d.normal(0.7 * surprise, 0.05),
        "release",
    )
    schedule = r.EventSchedule(
        [release, r.ScheduledEvent(5.0, d.normal(0, 0.2), name="decision")]
    )
    realization = schedule.sample(0, 5, rng=rng)
    print("Scheduled event times and impacts:", realization.times, realization.marks)
    print(
        "Conditional release impact quantiles:",
        release.impact_distribution(draws=3000, rng=rng).quantile([0.05, 0.5, 0.95]),
    )

    intensity = r.PiecewiseConstantIntensity([2, 3], [0.2, 1.5, 0.2])
    arrivals = r.PoissonProcess(intensity)
    jumps = r.MarkedPointProcess(arrivals, d.normal(-0.1, 0.3))
    print(
        "Expected event count and P(at least one):",
        arrivals.count_distribution(0, 5).mean(),
        arrivals.event_probability(0, 5),
    )
    print("Exact jump total mean/variance:", jumps.moments(0, 5))
    print("Realized marked events:", jumps.sample(0, 5, rng=rng))
    hawkes = r.ExponentialHawkes(0.3, 0.6, 1.2)
    events = hawkes.sample(0, 20, rng=rng)
    print("Self-exciting arrivals:", events)
    print(
        "Conditional point-process log likelihood:",
        hawkes.log_likelihood(events, 0, 20),
    )


if __name__ == "__main__":
    main()
