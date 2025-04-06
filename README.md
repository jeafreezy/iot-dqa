# IoT-DQA

The IoT-DQA library is a Python package designed to streamline Data Quality Assessment (DQA) for IoT time-series data. It provides robust tools for validating and analyzing IoT data streams, ensuring reliable data for downstream applications.


---

**Documentation**: <a href="https://jeafreezy.github.io/iot-dqa" target="_blank">https://jeafreezy.github.io/iot-dqa/</a>

**Source Code**: <a href="https://github.com/jeafreezy/iot-dqa" target="_blank">https://github.com/jeafreezy/iot-dqa</a>

---


## Key Features

- **Optimized Performance**: Handles large-scale IoT datasets efficiently, powered by the high-performance Polars library.
- **Streamlined Validation**: Simplifies the process of validating and analyzing IoT data streams.
- **Custom Metrics**: Tailor metrics to meet specific requirements.
- **Comprehensive Scoring**: Generates detailed data quality scores across multiple dimensions.
- **Seamless Integration**: Export results in formats like CSV and GeoJSON for easy integration with other tools.

## Dimensions of Data Quality
- **Validity**: Verifies data adherence to expected formats and ranges.
- **Accuracy**: Identifies and quantifies outliers using advanced techniques.
- **Completeness**: Evaluates the presence of missing or null values.
- **Timeliness**: Measures data arrival punctuality based on timestamps.

### Note:
- Designed for cumulative time-series data (e.g., utility consumption).
- Sample data is available in `tests/test_data.csv`.

## Installation
```bash
pip install iot_dqa
```

## Quick Start

### Example: Calculate Data Quality Score for IoT time-series data
```python
from iot_dqa import DataQualityScore, Dimension, OutlierDetectionAlgorithm, CompletenessStrategy

# Initialize and compute the Data Quality Score
dq_score = DataQualityScore(
    "./data/sample_data.csv",
    multiple_devices=True,
    dimensions=[
        Dimension.VALIDITY.value,
        Dimension.ACCURACY.value,
        Dimension.COMPLETENESS.value,
        Dimension.TIMELINESS.value,
    ],
    col_mapping={
        "latitude": "LAT",
        "longitude": "LONG",
        "date": "DATE",
        "value": "VALUE",
        "id": "DEVICE_ID",
    },
    metrics_config={
        "timeliness": {"iat_method": "min"},
        "accuracy": {
            "ensemble": True,
            "strategy": "validity",
            "algorithms": [
                OutlierDetectionAlgorithm.IF.value,
                OutlierDetectionAlgorithm.IQR.value,
                OutlierDetectionAlgorithm.MAD.value,
            ],
        },
        "completeness_strategy": CompletenessStrategy.ONLY_NULLS.value,
    },
).compute_score(
    weighting_mechanism="ahp",
    output_format="geojson",
    output_path="./output",
    ahp_weights={
        Dimension.VALIDITY.value: 0.3,
        Dimension.ACCURACY.value: 0.3,
        Dimension.COMPLETENESS.value: 0.3,
        Dimension.TIMELINESS.value: 0.1,
    },
)

print("Data Quality Score computed successfully!")
```

## Configuration Overview

| Configuration         | Attribute                  | Default Value       | Description                                                                 |
|-----------------------|----------------------------|---------------------|-----------------------------------------------------------------------------|
| **Isolation Forest**  | `n_estimators`            | `100`               | Number of trees in the forest.                                             |
|                       | `max_samples`             | `0.8`               | Proportion of samples for training each base estimator.                    |
|                       | `contamination`           | `0.1`               | Proportion of outliers in the dataset.                                     |
|                       | `max_features`            | `1`                 | Number of features for training each base estimator.                       |
|                       | `random_state`            | `42`                | Random seed for reproducibility.                                           |
| **Accuracy**          | `ensemble`                | `True`              | Use ensemble methods for accuracy.                                         |
|                       | `mad_threshold`           | `3`                 | Threshold for Median Absolute Deviation (MAD).                             |
|                       | `optimize_iqr_with_optuna`| `True`              | Enable IQR optimization using Optuna.                                      |
|                       | `iqr_optuna_q1_max`       | `0.5`               | Maximum value for Q1 in IQR optimization.                                  |
|                       | `iqr_optuna_q3_min`       | `0.5`               | Minimum value for Q3 in IQR optimization.                                  |
|                       | `iqr_optuna_q3_max`       | `1`                 | Maximum value for Q3 in IQR optimization.                                  |
|                       | `algorithms`              | All algorithms      | List of outlier detection algorithms.                                      |
|                       | `strategy`                | `NONE`              | Strategy for accuracy computation.                                         |
| **Timeliness**        | `iat_method`              | `min`               | Method to calculate inter-arrival time.                                    |
| **Completeness**      | `completeness_strategy`   | `ONLY_NULLS`        | Strategy for handling completeness.                                        |

For more details on configuration, refer to the [documentation](https://jeafreezy.github.io/iot-dqa/api/utils/configs/).

## Documentation
Visit the [documentation](https://jeafreezy.github.io/iot-dqa) for comprehensive details.

## Contributing
Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
