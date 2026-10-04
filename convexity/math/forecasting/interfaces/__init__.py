"""Forecasting extension contracts and mathematical result objects."""
from .base import ConvergenceError, Diagnostics, FittedForecaster, ForecastAdapter, Forecaster, fit, load_forecaster
from .data import Covariates, ForecastHorizon, TimeSeries
from .result import ForecastResult

__all__ = ['ConvergenceError', 'Diagnostics', 'FittedForecaster', 'ForecastAdapter', 'Forecaster',
           'fit', 'load_forecaster', 'Covariates', 'ForecastHorizon', 'TimeSeries', 'ForecastResult']
