"""Shared validation and explicit convention bridge for financial families."""

from ._validation import array, scalar, integer, choice
from ._backends.quantlib import qdate, day_count, evaluation_date, option_type
from dataclasses import dataclass


def positive(value, name):
    return scalar(value, name, minimum=0, strict=True)


def unit_interval(value, name):
    value = scalar(value, name, minimum=0)
    if value > 1:
        raise ValueError(f"{name} must be <= 1")
    return value


__all__ = [
    "array",
    "scalar",
    "integer",
    "choice",
    "qdate",
    "day_count",
    "evaluation_date",
    "option_type",
    "positive",
    "unit_interval",
]


@dataclass(frozen=True)
class ResultRecord:
    """Read-only keyed access for typed result records (legacy dict compatibility)."""

    def __getitem__(self, key):
        from dataclasses import fields

        if key not in {field.name for field in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def keys(self):
        from dataclasses import fields

        return tuple(field.name for field in fields(self))

    def items(self):
        return tuple((key, self[key]) for key in self.keys())
