import time
from logger import logger
from dataclasses import dataclass, field
from enums import (
    FrequencyCalculationMethod,
    Offset,
    OutlierDetectionAlgorithm,
)


@dataclass
class AccuracyConfig:
    ensemble: bool = True
    algorithms: list[OutlierDetectionAlgorithm] = field(
        default_factory=lambda: [x.value for x in OutlierDetectionAlgorithm]
    )

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
