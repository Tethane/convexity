import pytest
from convexity.math.curves import flat_curve
from convexity.math.pricing import BlackScholesModel


@pytest.fixture
def valuation():
    return "2026-01-01"


@pytest.fixture
def discount(valuation):
    return flat_curve(valuation, 0.05)


@pytest.fixture
def dividend(valuation):
    return flat_curve(valuation, 0.02)


@pytest.fixture
def bsm(discount, dividend):
    return BlackScholesModel(100.0, discount, dividend, 0.2)
