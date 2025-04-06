
# Example Usage

## Example 1: Compute a Few Metrics

To compute specific metrics, specify the dimensions you want:

```python
from iot_dqa.utils.enums import Dimension

dqs = DataQualityScore(
    file_path="path/to/your/data.csv",
    col_mapping={
        "date": "date_column_name",
        "value": "value_column_name",
    },
    dimensions=[Dimension.VALIDITY.value, Dimension.TIMELINESS.value],
)

metrics = dqs.compute_metrics()
print(metrics)
```

## Example 2: Adjust Configuration

You can adjust the configuration for metrics computation:

```python
from iot_dqa.utils.configs import MetricsConfig, AccuracyConfig

custom_config = MetricsConfig(
    accuracy=AccuracyConfig(ensemble=True, algorithms=["z_score", "iqr"])
)

dqs = DataQualityScore(
    file_path="path/to/your/data.csv",
    col_mapping={
        "date": "date_column_name",
        "value": "value_column_name",
    },
    metrics_config=custom_config,
)

metrics = dqs.compute_metrics()
print(metrics)
```

## Example 3: Compute Score

To compute the overall data quality score:

```python
from iot_dqa.utils.enums import WeightingMechanism, OutputFormat

dqs = DataQualityScore(
    file_path="path/to/your/data.csv",
    col_mapping={
        "date": "date_column_name",
        "value": "value_column_name",
        "id": "device_id_column_name",
    },
)

scores = dqs.compute_score(
    weighting_mechanism=WeightingMechanism.EQUAL.value,
    output_format=OutputFormat.CSV.value,
    output_path="./output",
)
print(scores)
```

## Example 4: Export Score to File/GeoJSON/CSV

To export the computed scores to a file:

```python
dqs = DataQualityScore(
    file_path="path/to/your/data.csv",
    col_mapping={
        "date": "date_column_name",
        "value": "value_column_name",
        "id": "device_id_column_name",
    },
)

scores = dqs.compute_score(
    output_format="geojson",  # Options: "csv", "geojson"
    output_path="./output",
    export=True,
)
print("Scores exported successfully.")
```

## Example 5: AHP Weighting Example

To compute the data quality score using AHP (Analytic Hierarchy Process) weighting:

```python
from iot_dqa.utils.enums import WeightingMechanism

dqs = DataQualityScore(
    file_path="path/to/your/data.csv",
    col_mapping={
        "date": "date_column_name",
        "value": "value_column_name",
        "id": "device_id_column_name",
    },
)

ahp_weights = {
    "validity": 0.4,
    "accuracy": 0.3,
    "completeness": 0.2,
    "timeliness": 0.1,
}

scores = dqs.compute_score(
    weighting_mechanism=WeightingMechanism.AHP.value,
    ahp_weights=ahp_weights,
    output_format="csv",
    output_path="./output",
)
print(scores)
```

## Example 6: Isolation Forest for Outlier Detection

```python
from iot_dqa.utils.configs import MetricsConfig, AccuracyConfig
from sklearn.ensemble import IsolationForest

# Define custom metrics configuration with Isolation Forest
custom_metrics_config = MetricsConfig(
    accuracy=AccuracyConfig(
        ensemble=False,
        algorithms=["if"],
        isolation_forest={"n_estimators": 100, "max_samples": "auto", "random_state": 42}
    )
)

# Initialize DataQualityScore with Isolation Forest configuration
dqs = DataQualityScore(
    file_path="data/sample.csv",
    col_mapping={"date": "timestamp", "value": "sensor_value", "id": "device_id"},
    metrics_config=custom_metrics_config,
    dimensions=["accuracy"]
)

# Compute metrics
metrics = dqs.compute_metrics()
print(metrics)
```

## Example 7: Timeliness with Custom Inter-Arrival Time Method

```python
from iot_dqa.utils.configs import MetricsConfig, TimelinessConfig
from iot_dqa.utils.enums import FrequencyCalculationMethod

# Define custom metrics configuration for timeliness
custom_metrics_config = MetricsConfig(
    timeliness=TimelinessConfig(
        iat_method=FrequencyCalculationMethod.MODE.value
    )
)

# Initialize DataQualityScore with custom timeliness configuration
dqs = DataQualityScore(
    file_path="data/sample.csv",
    col_mapping={"date": "timestamp", "value": "sensor_value", "id": "device_id"},
    metrics_config=custom_metrics_config,
    dimensions=["timeliness"],
    multiple_devices=True
)

# Compute metrics
metrics = dqs.compute_metrics()
print(metrics)
```

## Example 8: Inter-Quartile Range (IQR) with Optuna Optimization

```python
from iot_dqa.utils.configs import MetricsConfig, AccuracyConfig

# Define custom metrics configuration with IQR and Optuna optimization
custom_metrics_config = MetricsConfig(
    accuracy=AccuracyConfig(
        ensemble=False,
        algorithms=["iqr"],
        optimize_iqr_with_optuna=True,
        iqr_optuna_trials=50,
        iqr_optuna_q1_min=0.1,
        iqr_optuna_q1_max=0.3,
        iqr_optuna_q3_min=0.7,
        iqr_optuna_q3_max=0.9
    )
)

# Initialize DataQualityScore with IQR configuration
dqs = DataQualityScore(
    file_path="data/sample.csv",
    col_mapping={"date": "timestamp", "value": "sensor_value", "id": "device_id"},
    metrics_config=custom_metrics_config,
    dimensions=["accuracy"]
)

# Compute metrics
metrics = dqs.compute_metrics()
print(metrics)
```

