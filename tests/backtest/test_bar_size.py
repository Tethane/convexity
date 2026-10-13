import pytest

from convexity.backtest.bar_size import parse_bar_size


@pytest.mark.parametrize(('value', 'label'), [('7m', '7 mins'), (' 17 minutes ', '17 mins'),
    ('90sec', '90 secs'), ('1.5 min', '90 secs'), ('2.5h', '150 mins'), ('60 seconds', '1 min'),
    ('1d', '1 day'), ('3 weeks', '3 weeks'), ('2mo', '2 months'), ('1yr', '1 year')])
def test_duration_aliases(value, label):
    assert parse_bar_size(value).label == label


@pytest.mark.parametrize('value', ['', None, 1, [], '0 mins', '-3 mins', 'nan min',
    '1 tick', '0.5 secs', '1.5 days', '1.1 seconds'])
def test_invalid_or_unrepresentable_durations(value):
    with pytest.raises(ValueError):
        parse_bar_size(value)
