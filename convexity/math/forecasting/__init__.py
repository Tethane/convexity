"""Generic forecasting, explicit uncertainty, and causal temporal evaluation.

Only NumPy loads on family import; optional/model dependencies load on use.
"""
from .builtins.specialist import load_panel_forecaster
from . import builtins, interfaces
from .builtins import (
    DiffusionForecaster, ChronosForecaster, FoundationForecaster, GenerativeForecaster,
    SktimeForecaster, TimesFMForecaster, BaselineForecaster, DriftForecaster,
    MovingMeanForecaster, NaiveForecaster, SeasonalMeanForecaster, SeasonalNaiveForecaster,
    ARForecaster, ARIMAForecaster, ARMAForecaster, DynamicFactorForecaster, ETSForecaster,
    ExponentialSmoothingForecaster, HoltForecaster, HoltWintersForecaster, SARIMAForecaster,
    SARIMAXForecaster, StatsmodelsForecaster, StructuralForecaster, VARForecaster,
    VARMAXForecaster, VECMForecaster, EnsembleForecaster, StackingForecaster, ensemble,
    SequenceSchema, TorchForecaster, RegressionForecaster,
)
from .interfaces import (
    ConvergenceError, Diagnostics, FittedForecaster, ForecastAdapter, Forecaster, fit,
    load_forecaster, Covariates, ForecastHorizon, TimeSeries, ForecastResult,
)
from .artifacts import load_portable, save_portable, register_artifact_codec
from . import metrics
from .diagnostics import covariate_sensitivity, residual_diagnostics
from .evaluation import (
    BacktestResult, ForecastRecord, SelectionResult, backtest, compare, grid_search,
    select_model,
)

from .builtins import (
    ConditionalGenerativeForecaster, FlowForecaster, VAEForecaster, AutoregressiveForecaster,
    ScoreForecaster, Panel, NeuralForecastForecaster, GlobalFittedForecaster, TFTForecaster,
    PatchTSTForecaster, NBEATSForecaster, NHiTSForecaster, EncoderDecoderForecaster,
    OUForecaster, BootstrapForecaster, Checkpoint, download_checkpoint, load_pretrained,
    Chronos2Forecaster, TimesFM25Forecaster,
)
from .interfaces.data import calendar_horizon

__all__ = [
    'load_panel_forecaster',
    'load_portable', 'save_portable', 'register_artifact_codec',
    'ConditionalGenerativeForecaster', 'FlowForecaster', 'VAEForecaster',
    'AutoregressiveForecaster', 'ScoreForecaster', 'Panel', 'NeuralForecastForecaster',
    'GlobalFittedForecaster', 'TFTForecaster', 'PatchTSTForecaster', 'NBEATSForecaster',
    'NHiTSForecaster', 'EncoderDecoderForecaster', 'OUForecaster', 'BootstrapForecaster',
    'Checkpoint', 'download_checkpoint', 'load_pretrained', 'Chronos2Forecaster',
    'TimesFM25Forecaster', 'calendar_horizon', 'builtins', 'interfaces', 'metrics',
    'covariate_sensitivity', 'residual_diagnostics', 'BacktestResult', 'ForecastRecord',
    'SelectionResult', 'backtest', 'compare', 'grid_search', 'select_model',
    'DiffusionForecaster', 'ChronosForecaster', 'FoundationForecaster', 'GenerativeForecaster',
    'SktimeForecaster', 'TimesFMForecaster', 'BaselineForecaster', 'DriftForecaster',
    'MovingMeanForecaster', 'NaiveForecaster', 'SeasonalMeanForecaster',
    'SeasonalNaiveForecaster', 'ARForecaster', 'ARIMAForecaster', 'ARMAForecaster',
    'DynamicFactorForecaster', 'ETSForecaster', 'ExponentialSmoothingForecaster',
    'HoltForecaster', 'HoltWintersForecaster', 'SARIMAForecaster', 'SARIMAXForecaster',
    'StatsmodelsForecaster', 'StructuralForecaster', 'VARForecaster', 'VARMAXForecaster',
    'VECMForecaster', 'EnsembleForecaster', 'StackingForecaster', 'ensemble', 'SequenceSchema',
    'TorchForecaster', 'RegressionForecaster', 'ConvergenceError', 'Diagnostics',
    'FittedForecaster', 'ForecastAdapter', 'Forecaster', 'fit', 'load_forecaster', 'Covariates',
    'ForecastHorizon', 'TimeSeries', 'ForecastResult',
]

# Structural callback contracts are public; no provider inheritance is required.
from .builtins.adapters import ForecastFunction, PathSampler
from .builtins.regression import RegressionEstimator
from .builtins.neural import ForecastDecoder
__all__ += ["ForecastFunction", "PathSampler", "RegressionEstimator", "ForecastDecoder"]
