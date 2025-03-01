import enum


class Dimension(enum.Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"


class OutlierDetectionAlgorithm(enum.Enum):
    IQR = "IQR"
    """Interquatile Range"""
    MAD = "MAD"
    """Median Absolute Deviation"""
    IF = "IF"
    """Isolation Forest"""


class Offset(enum.Enum):
    NS = "1ns"
    US = "1us"
    MS = "1ms"
    S = "1s"
    M = "1m"
    H = "1h"
    D = "1d"
    W = "1w"
    MO = "1mo"
    Q = "1q"
    Y = "1y"


class FrequencyCalculationMethod(enum.Enum):
    MIN = "min"
    """Minimum Inter Arrival Time (IAT)."""
    MODE = "mode"
    """Mode of Inter Arrival Time (IAT)."""
