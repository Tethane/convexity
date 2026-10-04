"""Working forecasting specifications with lazy numerical engines."""
from .specialist import load_panel_forecaster
from .adapters import (
    ChronosForecaster, FoundationForecaster, GenerativeForecaster, SktimeForecaster,
    TimesFMForecaster,
)
from .baselines import (
    BaselineForecaster, DriftForecaster, MovingMeanForecaster, NaiveForecaster,
    SeasonalMeanForecaster, SeasonalNaiveForecaster,
)
from .classical import (
    ARForecaster, ARIMAForecaster, ARMAForecaster, DynamicFactorForecaster, ETSForecaster,
    ExponentialSmoothingForecaster, HoltForecaster, HoltWintersForecaster, SARIMAForecaster,
    SARIMAXForecaster, StatsmodelsForecaster, StructuralForecaster, VARForecaster,
    VARMAXForecaster, VECMForecaster,
)
from .generative import DiffusionForecaster
from .composition import EnsembleForecaster, StackingForecaster, ensemble
from .neural import SequenceSchema, TorchForecaster
from .regression import RegressionForecaster

from .generative import (
    ConditionalGenerativeForecaster, FlowForecaster, VAEForecaster, AutoregressiveForecaster,
    ScoreForecaster,
)
from .specialist import (
    Panel, NeuralForecastForecaster, GlobalFittedForecaster, TFTForecaster, PatchTSTForecaster,
    NBEATSForecaster, NHiTSForecaster, EncoderDecoderForecaster,
)
from .elapsed import OUForecaster
from .uncertainty import BootstrapForecaster
from .pretrained import (
    Checkpoint, download_checkpoint, load_pretrained, Chronos2Forecaster, TimesFM25Forecaster,
)

__all__ = [
    'load_panel_forecaster',
    'ConditionalGenerativeForecaster', 'FlowForecaster', 'VAEForecaster',
    'AutoregressiveForecaster', 'ScoreForecaster', 'Panel', 'NeuralForecastForecaster',
    'GlobalFittedForecaster', 'TFTForecaster', 'PatchTSTForecaster', 'NBEATSForecaster',
    'NHiTSForecaster', 'EncoderDecoderForecaster', 'OUForecaster', 'BootstrapForecaster',
    'Checkpoint', 'download_checkpoint', 'load_pretrained', 'Chronos2Forecaster',
    'TimesFM25Forecaster', 'DiffusionForecaster', 'ChronosForecaster', 'FoundationForecaster',
    'GenerativeForecaster', 'SktimeForecaster', 'TimesFMForecaster', 'BaselineForecaster',
    'DriftForecaster', 'MovingMeanForecaster', 'NaiveForecaster', 'SeasonalMeanForecaster',
    'SeasonalNaiveForecaster', 'ARForecaster', 'ARIMAForecaster', 'ARMAForecaster',
    'DynamicFactorForecaster', 'ETSForecaster', 'ExponentialSmoothingForecaster',
    'HoltForecaster', 'HoltWintersForecaster', 'SARIMAForecaster', 'SARIMAXForecaster',
    'StatsmodelsForecaster', 'StructuralForecaster', 'VARForecaster', 'VARMAXForecaster',
    'VECMForecaster', 'EnsembleForecaster', 'StackingForecaster', 'ensemble', 'SequenceSchema',
    'TorchForecaster', 'RegressionForecaster',
]
