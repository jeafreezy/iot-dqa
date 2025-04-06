import pytest
from iot_dqa.utils.enums import Dimension, WeightingMechanism, OutputFormat
from iot_dqa.dq_score import DataQualityScore
from iot_dqa.utils.exceptions import (
    InsufficientDataException,
    InvalidDimensionException,
    InvalidColumnMappingException,
)


@pytest.fixture
def setup_data():
    return {
        "valid_col_mapping": {
            "latitude": "OGI_LAT",
            "longitude": "OGI_LONG",
            "date": "TAG_VALUE_DATE",
            "value": "TAG_VALUE_RAW",
            "id": "DEVICE_ID",
        },
        "valid_metrics_config": {
            "timeliness": {"iat_method": "min"},
            "accuracy": {
                "ensemble": True,
                "strategy": "validity",
                "algorithms": ["if", "iqr", "mad"],
            },
            "completeness_strategy": "only_nulls",
        },
        "valid_dimensions": [
            Dimension.VALIDITY.value,
            Dimension.ACCURACY.value,
            Dimension.COMPLETENESS.value,
            Dimension.TIMELINESS.value,
        ],
        "file_path": "./tests/test_data.csv",
    }


def test_data_loader_valid_file(setup_data):
    print(setup_data["file_path"])
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    df = dq._data_loader()
    assert df.shape == (63, 25)
    for col in [
        "OGI_LAT",
        "OGI_LONG",
        "TAG_VALUE_DATE",
        "TAG_VALUE_RAW",
        "DEVICE_ID",
    ]:
        assert col in df.columns


def test_validate_col_mapping_valid(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    df_cols = ["OGI_LAT", "OGI_LONG", "TAG_VALUE_DATE", "TAG_VALUE_RAW", "DEVICE_ID"]
    assert dq._validate_col_mapping(df_cols) is None


def test_validate_col_mapping_missing_required(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    df_cols = ["OGI_LAT", "OGI_LONG"]
    with pytest.raises(InvalidColumnMappingException):
        dq._validate_col_mapping(df_cols)


def test_validate_records_valid(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    assert dq._validate_records(100) is None


def test_validate_records_insufficient(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    with pytest.raises(InsufficientDataException):
        dq._validate_records(10)


def test_validate_dimensions_valid(setup_data):
    dq = DataQualityScore(
        setup_data["file_path"],
        setup_data["valid_col_mapping"],
        dimensions=setup_data["valid_dimensions"],
    )
    assert dq._validate_dimensions() is None


def test_validate_dimensions_invalid(setup_data):
    dq = DataQualityScore(
        setup_data["file_path"],
        setup_data["valid_col_mapping"],
        dimensions=["invalid_dimension"],
    )
    with pytest.raises(InvalidDimensionException):
        dq._validate_dimensions()


def test_validate_weighting_mechanism_valid(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    wm = dq._validate_weighting_mechanism(WeightingMechanism.EQUAL.value)
    assert wm == WeightingMechanism.EQUAL


def test_validate_weighting_mechanism_invalid(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    with pytest.raises(InvalidDimensionException):
        dq._validate_weighting_mechanism("invalid_mechanism")


def test_validate_output_format_valid(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    of = dq._validate_output_format(OutputFormat.CSV.value)
    assert of == OutputFormat.CSV


def test_validate_output_format_invalid(setup_data):
    dq = DataQualityScore(setup_data["file_path"], setup_data["valid_col_mapping"])
    with pytest.raises(InvalidDimensionException):
        dq._validate_output_format("invalid_format")


def test_validate_ahp_weights_valid(setup_data):
    dq = DataQualityScore(
        setup_data["file_path"],
        setup_data["valid_col_mapping"],
        dimensions=setup_data["valid_dimensions"],
    )
    ahp_weights = {
        Dimension.VALIDITY.value: 0.3,
        Dimension.ACCURACY.value: 0.3,
        Dimension.COMPLETENESS.value: 0.3,
        Dimension.TIMELINESS.value: 0.1,
    }
    assert dq._validate_ahp_weights(ahp_weights) is None


def test_validate_ahp_weights_invalid_sum(setup_data):
    dq = DataQualityScore(
        setup_data["file_path"],
        setup_data["valid_col_mapping"],
        dimensions=setup_data["valid_dimensions"],
    )
    ahp_weights = {
        Dimension.VALIDITY.value: 0.4,
        Dimension.ACCURACY.value: 0.4,
        Dimension.COMPLETENESS.value: 0.4,
        Dimension.TIMELINESS.value: 0.1,
    }
    with pytest.raises(InvalidDimensionException):
        dq._validate_ahp_weights(ahp_weights)


def test_integration_full_flow(setup_data):
    dq = DataQualityScore(
        setup_data["file_path"],
        setup_data["valid_col_mapping"],
        dimensions=setup_data["valid_dimensions"],
    )

    # Validate AHP weights
    ahp_weights = {
        Dimension.VALIDITY.value: 0.3,
        Dimension.ACCURACY.value: 0.3,
        Dimension.COMPLETENESS.value: 0.3,
        Dimension.TIMELINESS.value: 0.1,
    }

    # Simulate scoring process
    score = dq.compute_score(
        weighting_mechanism=WeightingMechanism.EQUAL.value,
        output_format=OutputFormat.CSV.value,
        ahp_weights=ahp_weights,
        export=False,
    )
    assert isinstance(score, dict)
    assert "overall_scores" in score
    assert "general_scores" in score
