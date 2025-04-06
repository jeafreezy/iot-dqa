import time
from typing import Union


from iot_dqa.utils.logger import logger
from dataclasses import dataclass, field
from iot_dqa.utils.enums import (
    AccuracyStrategy,
    CompletenessStrategy,
    FrequencyCalculationMethod,
    OutlierDetectionAlgorithm,
)


@dataclass
class IsolationForestConfig:
    """
    Configuration class for Isolation Forest settings.
    Attributes:
        n_estimators (int): Number of trees in the forest. Default is 100.
        max_samples (float): Number of samples to draw to train each base estimator. Default is 0.8.
        contamination (float): Proportion of outliers in the data set. Default is 0.1.
        max_features (int): Number of features to draw to train each base estimator. Default is 1.
    """

    n_estimators: int = 100
    max_samples: float = 0.8
    contamination: Union[float, str] = 0.1
    max_features: int = 1
    random_state: int = 42


@dataclass
class AccuracyConfig:
    """
    Configuration class for completeness settings.
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
    iqr_optuna_q1_min: Union[int, None] = 0
    """Minimum value for the first quartile (Q1) in IQR optimization. Default is 0.0."""
    iqr_optuna_q1_max: Union[float, None] = 0.5
    """Maximum value for the first quartile (Q1) in IQR optimization. Default is 0.5."""
    iqr_optuna_q3_min: Union[float, None] = 0.5
    """Minimum value for the third quartile (Q3) in IQR optimization. Default is 0.5."""
    iqr_optuna_q3_max: Union[int, None] = 1
    """Maximum value for the third quartile (Q3) in IQR optimization. Default is 1.0."""

    algorithms: list[OutlierDetectionAlgorithm] = field(
        default_factory=lambda: [x.value for x in OutlierDetectionAlgorithm]
    )
    """List of outlier detection algorithms to be used. Default is all values of OutlierDetectionAlgorithm."""
    strategy: AccuracyStrategy = AccuracyStrategy.NONE.value
    """Determine the approach to use for the accuracy computation."""
    isolation_forest: IsolationForestConfig = field(
        default_factory=IsolationForestConfig
    )
    """Configuration for Isolation Forest settings."""

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

        if not isinstance(self.optimize_iqr_with_optuna, bool):
            raise ValueError(
                f"IQR optimization must be valid boolean. Provided: {self.optimize_iqr_with_optuna}"
            )
        if not isinstance(self.mad_threshold, int):
            raise ValueError(
                f"MAD threshold must be valid boolean. Provided: {self.mad_threshold}"
            )
        if not isinstance(self.iqr_optuna_trials, int):
            raise ValueError(
                f"IQR optuna trial must be valid integer. Provided: {self.iqr_optuna_trials}"
            )

        if not isinstance(self.iqr_optuna_q1_min, int):
            raise ValueError(
                f"IQR optuna Q1 min must be valid integer. Provided: {self.iqr_optuna_q1_min}"
            )

        if not isinstance(self.iqr_optuna_q1_max, float):
            raise ValueError(
                f"IQR optuna Q1 max must be valid float. Provided: {self.iqr_optuna_q1_max}"
            )

        if not isinstance(self.iqr_optuna_q3_min, float):
            raise ValueError(
                f"IQR optuna Q1 min must be valid float. Provided: {self.iqr_optuna_q3_min}"
            )

        if not isinstance(self.iqr_optuna_q3_max, int):
            raise ValueError(
                f"IQR optuna Q1 max must be valid integer. Provided: {self.iqr_optuna_q3_max}"
            )

        if self.iqr_optuna_q3_max > 1:
            raise ValueError(
                f"IQR Q3 max must be less than or equal to 1. Provided: {self.iqr_optuna_q3_max}"
            )
        if self.iqr_optuna_q1_min < 0:
            raise ValueError(
                f"IQR Q1 min must be greater than or equal to 0. Provided: {self.iqr_optuna_q1_min}"
            )


@dataclass
class TimelinessConfig:
    """
    Configuration class for timeliness settings.
    Attributes:
        frequency (str): Frequency of the data. Default is "1H".
        iat_method (str): Method to calculate inter-arrival time. Default is "mean".
    """

    iat_method: FrequencyCalculationMethod = FrequencyCalculationMethod.MIN.value
    """Method to calculate inter-arrival time. Default is 'min'."""

    def __post_init__(self):
        if not (self.iat_method):
            raise ValueError(
                "At least one of 'frequency_unit' or 'frequency_unit' or 'iat_method' must be provided."
            )

        if not isinstance(self.iat_method, str):
            raise ValueError(
                f"IAT method must be valid string. Provided: {self.iat_method}"
            )


@dataclass
class MetricsConfig:
    accuracy: AccuracyConfig = field(default_factory=AccuracyConfig)
    timeliness: TimelinessConfig = field(default_factory=TimelinessConfig)
    completeness_strategy: CompletenessStrategy = CompletenessStrategy.ONLY_NULLS.value


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
