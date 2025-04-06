from dataclasses import asdict
import math
import os
from typing import Union

from iot_dqa.dimensions import Accuracy, Completeness, Timeliness, Validity

from iot_dqa.utils.configs import AccuracyConfig, MetricsConfig, TimelinessConfig

from iot_dqa.utils.enums import (
    ColumnMappingColumnName,
    CompletenessStrategy,
    Dimension,
    OutputFormat,
    WeightingMechanism,
)
import polars as pl
from iot_dqa.utils.exceptions import (
    InsufficientDataException,
    InvalidDimensionException,
    InvalidFileException,
    InvalidColumnMappingException,
)
from iot_dqa.utils.logger import logger
import json


class DataQualityScore:
    def __init__(
        self,
        file_path: Union[str, None],
        col_mapping: dict[str, str],
        metrics_config: MetricsConfig = asdict(MetricsConfig()),
        dimensions: list[Dimension] = [x.value for x in Dimension],
        multiple_devices: bool = False,
    ):
        self.file_path = file_path
        self.col_mapping = col_mapping
        self.metrics_config = metrics_config
        self.dimensions = dimensions
        self.multiple_devices = multiple_devices

    def _validate_col_mapping(self, df_cols: list[str]) -> bool:
        required_cols = {
            ColumnMappingColumnName.DATE.value,
            ColumnMappingColumnName.VALUE.value,
        }

        logger.info("Validating column mappings...")
        # Validate required columns
        missing_required = required_cols - self.col_mapping.keys()
        if missing_required:
            raise InvalidColumnMappingException(
                f"The required columns: {', '.join(missing_required)} are not in the column mapping dictionary. Provide these keys and retry."
            )

        # Validate ID column if multiple devices is enabled
        if self.multiple_devices:
            if ColumnMappingColumnName.DEVICE_ID.value not in self.col_mapping.keys():
                raise InvalidColumnMappingException(
                    "'id' is required when 'multiple_devices' is enabled. Provide it and retry."
                )

        # Validate column mapping values
        invalid_values = [
            v for v in self.col_mapping.values() if not isinstance(v, str)
        ]
        if invalid_values:
            raise InvalidColumnMappingException(
                f"The following values should be strings: {', '.join(invalid_values)}"
            )

        # Validate that column mapping values exist in the dataframe columns
        missing_in_df = [v for v in self.col_mapping.values() if v not in df_cols]
        if missing_in_df:
            logger.error(
                f"The following columns are missing in the provided data: {', '.join(missing_in_df)}"
            )
            raise InvalidColumnMappingException(
                f"The following columns are missing in the provided data: {', '.join(missing_in_df)}"
            )
        logger.info("Column mapping validation completed without errrors...")
        return

    def _validate_records(self, df_shape: int):
        logger.info("Validating records...")
        if df_shape < 50:
            logger.error(
                f"The provided data ({df_shape}) records, is insufficient. At least 50 records are required in the CSV."
            )
            raise InsufficientDataException(
                "The provided data is insufficient. At least 50 records are required in the CSV."
            )
        logger.info("Record validation completed without errrors...")
        return

    def _validate_dimensions(self):
        logger.info("Validating dimensions...")
        supported_dimensions = [x.value for x in Dimension]
        if isinstance(self.dimensions, list):
            for dimension in self.dimensions:
                if dimension.lower() not in supported_dimensions:
                    logger.error(
                        f"The provided dimension: {dimension} is invalid. Only the following are supported:{supported_dimensions}"
                    )
                    raise InvalidDimensionException(
                        f"The provided dimension: {dimension} is invalid. Only the following are supported:{supported_dimensions}"
                    )
        logger.info("Dimension validation completed without errrors...")
        return

    def _validate_config(self):
        try:
            logger.info("Validating metrics configuration...")

            self.metrics_config = MetricsConfig(
                timeliness=TimelinessConfig(
                    **(
                        self.metrics_config.get("timeliness")
                        if self.metrics_config.get("timeliness")
                        else {}
                    )
                ),
                accuracy=AccuracyConfig(
                    **(
                        self.metrics_config.get("accuracy")
                        if self.metrics_config.get("accuracy")
                        else {}
                    )
                ),
                completeness_strategy=(
                    self.metrics_config.get("completeness_strategy")
                    if self.metrics_config.get("completeness_strategy")
                    else CompletenessStrategy.ONLY_NULLS.value
                ),
            )
            if self.metrics_config.accuracy.ensemble:
                if len(self.metrics_config.accuracy.algorithms) < 2:
                    raise InvalidDimensionException(
                        "At least two outlier detection algorithms are required when ensemble is enabled."
                    )
            else:
                if len(self.metrics_config.accuracy.algorithms) != 1:
                    raise InvalidDimensionException(
                        "Exactly one outlier detection algorithm is required when ensemble is not enabled."
                    )

            logger.info("Metrics configuration validation completed without errors...")

            return
        except Exception as e:
            logger.error(
                f"An error occured during metrics configuration validation. Provided metrics: {self.metrics_config} -> Error: {e} "
            )
            raise e

    def _data_loader(self) -> pl.DataFrame:
        """Load the data from the CSV file using Polars.

        Raises:
            InvalidFileException: Raised when an invalid file is provided.

        Returns:
            pl.DataFrame: The Polars DataFrame object of the file.
        """
        logger.info("Loading the data from the CSV file...")
        try:
            df = pl.read_csv(self.file_path, infer_schema_length=10000)
            logger.info("Data loaded successfully.")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Data columns: {df.columns}")
            logger.info(f"First few rows:\n{df.head()}")
            return df
        except Exception as e:
            logger.error(
                f"An error occurred while loading the data from the CSV file. Error: {e}"
            )
            raise InvalidFileException(
                f"The provided file is invalid. Ensure the path is valid and it is a valid CSV. Error: {e}"
            )

    def _validate_weighting_mechanism(
        self, weighting_mechanism: Union[WeightingMechanism, str]
    ) -> WeightingMechanism:
        """Validate the weighting mechanism.

        Args:
            weighting_mechanism (WeightingMechanism | str): The weighting mechanism to validate.

        Raises:
            InvalidDimensionException: Raised when an invalid weighting mechanism is provided.

        Returns:
            WeightingMechanism: The validated weighting mechanism.
        """
        logger.info("Validating the weighting mechanism...")
        if isinstance(weighting_mechanism, str):
            if weighting_mechanism not in [wm.value for wm in WeightingMechanism]:
                raise InvalidDimensionException(
                    f"The provided weighting mechanism: {weighting_mechanism} is invalid. Only the following are supported: {[wm.value for wm in WeightingMechanism]}"
                )
            return WeightingMechanism(weighting_mechanism)
        logger.info("Weighting mechanism validation completed without errors...")
        return weighting_mechanism

    def _validate_output_format(
        self, output_format: Union[OutputFormat, str]
    ) -> OutputFormat:
        """Validate the output format.

        Args:
            output_format (OutputFormat | str): The output format to validate.

        Raises:
            InvalidDimensionException: Raised when an invalid output format is provided.

        Returns:
            OutputFormat: The validated output format.
        """
        logger.info("Validating the output format...")
        if isinstance(output_format, str):
            if output_format not in [v.value for v in OutputFormat]:
                raise InvalidDimensionException(
                    f"The provided output format: {output_format} is invalid. Only the following are supported:{[v.value for v in OutputFormat]}"
                )
            return OutputFormat(output_format)
        logger.info("Output format validation completed without errors...")
        return output_format

    def _validate_output_path(self, output_path: str):
        """Validate the output path.

        Args:
            output_path (str): The output path to validate.

        Raises:
            InvalidDimensionException: Raised when an invalid output path is provided.
        """
        logger.info("Validating the output path...")
        if not isinstance(output_path, str):
            raise InvalidDimensionException(
                f"The provided output path: {output_path} is invalid. It should be a string."
            )
        logger.info("Output path validation completed without errors...")
        return

    def _validate_ahp_weights(self, ahp_weights: dict[str, float]):
        """Validate the AHP weights.

        Args:
            ahp_weights (dict[str, float]): The AHP weights to validate.

        Raises:
            InvalidDimensionException: Raised when invalid AHP weights are provided.
        """
        logger.info("Validating the AHP weights...")
        if not isinstance(ahp_weights, dict):
            raise InvalidDimensionException(
                f"The provided AHP weights: {ahp_weights} are invalid. They should be a dictionary."
            )
        if len(ahp_weights) != len(self.dimensions):
            raise InvalidDimensionException(
                f"The provided AHP weights: {ahp_weights} are invalid. The number of weights should match the number of dimensions."
            )

        if len(ahp_weights) > 4:
            raise InvalidDimensionException(
                f"The provided AHP weights: {ahp_weights} are invalid. A maximum of 4 weights is allowed."
            )
        if not math.isclose(sum(ahp_weights.values()), 1.0, rel_tol=1e-9):
            raise InvalidDimensionException(
                f"The provided AHP weights: {ahp_weights} are invalid. The weights must sum to 1."
            )
        logger.info("AHP weights validation completed without errors...")
        return

    def compute_metrics(self):
        # load file
        df = self._data_loader()
        # validations
        self._validate_col_mapping(df.columns)
        self._validate_records(df.shape[0])
        self._validate_dimensions()
        self._validate_config()

        # select the necessary columns, incase the csv has many columns
        df = df.select(list(self.col_mapping.values()))

        logger.info(f"Selected the necessary columns: {df.head()}")

        # sort the data by date and id
        if self.multiple_devices:
            df = df.sort(
                by=[
                    self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value],
                    self.col_mapping[ColumnMappingColumnName.DATE.value],
                ]
            )
        else:
            df = df.sort(by=self.col_mapping[ColumnMappingColumnName.DATE.value])

        df_metrics = df
        # based on the selected dimensions, instantiate the classes.
        if Dimension.VALIDITY.value in self.dimensions:
            logger.info("Computing validity metric...")

            df_metrics = Validity(
                df,
                col_mapping=self.col_mapping,
                multiple_devices=self.multiple_devices,
                config=self.metrics_config,
            ).compute_metric()

            logger.info("Validity metric completed...")
            logger.info(f"First few rows: {df_metrics.head()}")

        if Dimension.ACCURACY.value in self.dimensions:
            logger.info("Computing accuracy metric...")
            df_metrics = Accuracy(
                df_metrics,
                col_mapping=self.col_mapping,
                multiple_devices=self.multiple_devices,
                config=self.metrics_config,
            ).compute_metric()
            logger.info("Accuracy metric completed...")
            logger.info(f"First few rows: {df_metrics.head()}")

        if Dimension.COMPLETENESS.value in self.dimensions:
            logger.info("Computing completeness metric...")
            df_metrics = Completeness(
                df_metrics,
                col_mapping=self.col_mapping,
                multiple_devices=self.multiple_devices,
                config=self.metrics_config,
            ).compute_metric()
            logger.info("Completeness metric completed...")
            logger.info(f"First few rows: {df_metrics.head()}")

        if Dimension.TIMELINESS.value in self.dimensions:
            logger.info("Computing timeliness metric...")
            df_metrics = Timeliness(
                df_metrics,
                col_mapping=self.col_mapping,
                multiple_devices=self.multiple_devices,
                config=self.metrics_config,
            ).compute_metric()
            logger.info("Timeliness metric completed...")
            logger.info(f"First few rows: {df_metrics.head()}")

        return df_metrics

    def compute_score(
        self,
        weighting_mechanism: str = WeightingMechanism.EQUAL.value,
        output_format: Union[str, OutputFormat] = OutputFormat.CSV.value,
        output_path: str = "./output",
        ahp_weights: dict[str, float] = None,
        export: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Compute the score for each device and overall scores for all devices.

        Args:
            weighting_mechanism (str|WeightingMechanism, optional): The weighting mechanism to use for a single score. Defaults to "Equal".
            output_format (str|OutputFormat, optional): The output format for the scores. Defaults to "csv".
            output_path (str, optional): The path to save the output. Defaults to "./output".
            ahp_weights (dict[str, float], optional): Weights for AHP computation. Required if weighting_mechanism is "AHP".

        Returns:
            dict: A dictionary containing scores per device and overall scores.
        """
        logger.info("Computing the score for each device and overall scores...")
        # Ensure the metrics have been computed
        df_with_metrics = self.compute_metrics()

        # Validate the weighting mechanism
        weighting_mechanism = self._validate_weighting_mechanism(weighting_mechanism)
        # Validate the output format
        output_format = self._validate_output_format(output_format)
        # Validate the output path
        self._validate_output_path(output_path)
        # Validate the AHP weights if using AHP
        if weighting_mechanism == WeightingMechanism.AHP:
            self._validate_ahp_weights(ahp_weights)

        # Compute the score for each dimension for each device
        scores_per_device = {}
        for device_id, group in df_with_metrics.group_by(
            self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
        ):
            device_id = device_id[0]
            scores_per_device[device_id] = {
                dimension: group[dimension].mean()
                for dimension in self.dimensions
                if dimension in group.columns
            }

        logger.info(f"Scores per device: {scores_per_device}")

        # Compute the general score for each device based on the weighting mechanism
        general_scores = {}
        if weighting_mechanism == WeightingMechanism.EQUAL:
            equal_weight = 1 / len(self.dimensions)
            for device_id, scores in scores_per_device.items():
                general_scores[device_id] = sum(
                    scores[dim] * equal_weight for dim in scores
                )
        elif weighting_mechanism == WeightingMechanism.AHP:
            for device_id, scores in scores_per_device.items():
                general_scores[device_id] = sum(
                    scores[dim] * ahp_weights.get(dim, 0) for dim in scores
                )
        else:
            equal_weight = 1 / len(self.dimensions)
            for device_id, scores in scores_per_device.items():
                equal_score = sum(scores[dim] * equal_weight for dim in scores)
                ahp_score = sum(scores[dim] * ahp_weights.get(dim, 0) for dim in scores)
                general_scores[device_id] = (equal_score + ahp_score) / 2

        # Add the general score to the scores per device
        for device_id in scores_per_device:
            scores_per_device[device_id]["dqs"] = general_scores[device_id]

        logger.info(f"Scores per device with dqs: {scores_per_device}")

        # Compute overall scores for all devices
        overall_scores = {
            dimension: sum(scores[dimension] for scores in scores_per_device.values())
            / len(scores_per_device)
            for dimension in self.dimensions
        }
        overall_scores["dqs"] = sum(general_scores.values()) / len(general_scores)

        logger.info(f"Overall Data Quality Scores: {overall_scores}")

        # Ensure the output directory exists
        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            logger.error(
                f"An error occurred while creating the output directory: {output_path}. Error: {e}"
            )
            raise e
        if export:
            if output_format == OutputFormat.CSV:
                output_file_device = f"{output_path}/scores_per_device.csv"
                output_file_overall = f"{output_path}/overall_scores.csv"

                # Convert scores_per_device to a DataFrame for saving
                scores_per_device_df = pl.DataFrame(
                    [
                        {"device_id": device_id, **scores}
                        for device_id, scores in scores_per_device.items()
                    ]
                )
                scores_per_device_df.write_csv(output_file_device)

                # Save overall scores to a CSV file
                overall_scores_df = pl.DataFrame([overall_scores])
                overall_scores_df.write_csv(output_file_overall)

                logger.info(f"Scores per device saved to CSV at {output_file_device}")
                logger.info(f"Overall scores saved to CSV at {output_file_overall}")
            else:
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    df_with_metrics.filter(
                                        pl.col(self.col_mapping["id"]) == device_id
                                    )[self.col_mapping["longitude"]].to_list()[0],
                                    df_with_metrics.filter(
                                        pl.col(self.col_mapping["id"]) == device_id
                                    )[self.col_mapping["latitude"]].to_list()[0],
                                ],
                            },
                            "properties": {
                                "device_id": device_id,
                                **{
                                    dimension: scores_per_device[device_id].get(
                                        dimension, 0
                                    )
                                    for dimension in self.dimensions
                                },
                                "dqs": general_scores[device_id],
                            },
                        }
                        for device_id in scores_per_device
                    ],
                }
                output_file_device = f"{output_path}/scores_per_device.geojson"
                output_file_overall = f"{output_path}/overall_scores.json"
                with open(output_file_device, "w") as f:
                    json.dump(geojson_data, f)
                with open(output_file_overall, "w") as f:
                    json.dump(overall_scores, f)
                logger.info(
                    f"Scores per device saved to GeoJSON at {output_file_device}"
                )
                logger.info(f"Overall scores saved to JSON at {output_file_overall}")

        return {
            "scores_per_device": scores_per_device,
            "general_scores": general_scores,
            "overall_scores": overall_scores,
        }

    def __repr__(self):
        print("<DataQualityScore>")
