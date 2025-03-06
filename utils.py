import time
from typing import Union
from logger import logger
from dataclasses import dataclass, field
from enums import (
    FrequencyCalculationMethod,
    Offset,
    OutlierDetectionAlgorithm,
)


@dataclass
class AccuracyConfig:
    """
    Configuration class for accuracy settings in outlier detection.
    Methods:
        __post_init__(): Validates the provided algorithms and ensemble flag.
    """

    ensemble: bool = True
    """Flag to indicate if ensemble methods should be used. Default is True."""
    mad_threshold: int = 3
    """ Threshold for Median Absolute Deviation (MAD). Default is 3. Using 3 * STD as decribed in the literature."""
    optimize_iqr_with_optuna: bool = True
    """Flag to indicate if IQR optimization should be performed using optuna. Default is True."""
    iqr_optuna_trials: Union[int, None] = 10
    """10 trials when optimizing the IQR"""
    iqr_optuna_q1_min: Union[float, None] = 0.0
    """Minimum value for the first quartile (Q1) in IQR optimization. Default is 0.0."""
    iqr_optuna_q1_max: Union[float, None] = 0.5
    """Maximum value for the first quartile (Q1) in IQR optimization. Default is 0.5."""
    iqr_optuna_q3_min: Union[float, None] = 0.5
    """Minimum value for the third quartile (Q3) in IQR optimization. Default is 0.5."""
    iqr_optuna_q3_max: Union[float, None] = 1.0
    """Maximum value for the third quartile (Q3) in IQR optimization. Default is 1.0."""

    algorithms: list[OutlierDetectionAlgorithm] = field(
        default_factory=lambda: [x.value for x in OutlierDetectionAlgorithm]
    )
    """List of outlier detection algorithms to be used. Default is all values of OutlierDetectionAlgorithm."""

    def __post_init__(self):
        if not all(
            algo in OutlierDetectionAlgorithm._value2member_map_
            for algo in self.algorithms
        ):
            raise ValueError(
                f"All algorithms must be valid values of OutlierDetectionAlgorithm. Provided: {self.algorithms}"
            )
        if not isinstance(self.ensemble, bool):
            raise ValueError(
                f"Ensemble must be valid boolean. Provided: {self.ensemble}"
            )


@dataclass
class MetricsConfig:
    accuracy: AccuracyConfig
    frequency: Offset = None
    iat_method: FrequencyCalculationMethod = None

    def __post_init__(self):
        if self.frequency:
            if self.frequency not in Offset._value2member_map_:
                raise ValueError(
                    f"Invalid frequency: {self.frequency}. Must be one of {list(Offset._value2member_map_.keys())}"
                )

        if not (self.frequency or self.iat_method):
            raise ValueError(
                "At least one of 'frequency' or 'iat_method' must be provided."
            )


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete."
        )
        return result

    return wrapper
