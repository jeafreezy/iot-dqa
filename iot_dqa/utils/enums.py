import enum


class Dimension(enum.Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"


class WeightingMechanism(enum.Enum):
    EQUAL = "equal"
    AHP = "ahp"
    BOTH = "both"
    """Equal weighting or AHP weighting or both."""


class OutlierDetectionAlgorithm(enum.Enum):
    IQR = "IQR"
    """Interquatile Range"""
    MAD = "MAD"
    """Median Absolute Deviation"""
    IF = "IF"
    """Isolation Forest"""


class FrequencyCalculationMethod(enum.Enum):
    MIN = "min"
    """Minimum Inter Arrival Time (IAT)."""
    MODE = "mode"
    """Mode of Inter Arrival Time (IAT)."""


class OutputFormat(enum.Enum):
    CSV = "csv"
    """Comma Separated Values"""
    GEOJSON = "geojson"
    """JavaScript Object Notation"""


class CompletenessStrategy(enum.Enum):
    ONLY_NULLS = "nulls"
    """Consider missing/nulls as incomplete."""
    ACCURACY = "accuracy"
    """Consider only accurate values as complete."""
    TIMELINESS = "timeliness"
    """Checks the IAT of the device to compute the expected records vs the sent records."""


class ColumnMappingColumnName(enum.Enum):
    TIMESTAMP = "timestamp"
    DEVICE_ID = "id"
    VALUE = "value"
    LONGITUDE = "longitude"
    LATITUDE = "latitude"
    DATE = "date"


class ColumnName(enum.Enum):
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    IF_OUTLIERS = "IF_outliers"
    IQR_OUTLIERS = "IQR_outliers"
    MAD_OUTLIERS = "MAD_outliers"
    IAT = "iat"
    RAE = "rae"
    GOODNESS = "goodness"
    PENALTY = "penalty"
    EXPECTED_INTERVAL = "expected_interval"
    FILL_VALUE = "fill_value"


class AccuracyStrategy(enum.Enum):
    VALIDITY = "v"
    NONE = "none"
