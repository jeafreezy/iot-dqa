from dataclasses import asdict

from dimensions import Accuracy, Validity

from utils import AccuracyConfig, MetricsConfig

from enums import (
    Dimension,
    FrequencyCalculationMethod,
    OutlierDetectionAlgorithm,
)
import polars as pl
from exceptions import (
    InsufficientDataException,
    InvalidDimensionException,
    InvalidFileException,
    InvalidColumnMappingException,
)
from logger import logger


# completeness is default and no configurable. Look for nulls. Compute IAT and find empty records/missing dates not in the dataset.
# validity is constant. Check for drops etc and count them.
# final dqs config - AHP excel sheet if available, otherwise use equal
# make it a method
class DataQualityScore:
    def __init__(
        self,
        file_path: str,
        col_mapping: dict[str, str],
        metrics_config: MetricsConfig = asdict(
            MetricsConfig(
                iat_method=FrequencyCalculationMethod.MIN.value,
                accuracy=AccuracyConfig(),
            )
        ),
        dimensions: list[Dimension] = [x.value for x in Dimension],
        generate_dashboard: bool = False,
        multiple_devices: bool = False,
    ):
        self.file_path = file_path
        self.col_mapping = col_mapping
        self.generate_dashboard = generate_dashboard
        self.metrics_config = metrics_config
        self.dimensions = dimensions
        self.multiple_devices = multiple_devices

    def _validate_col_mapping(self, df_cols: list[str]) -> bool:
        required_cols = {"date": "value"}
        optional_cols = {"latitude", "longitude"}
        logger.info("Validating column mappings...")
        # Validate required columns
        missing_required = required_cols - self.col_mapping.keys()
        if missing_required:
            raise InvalidColumnMappingException(
                f"The required columns: {', '.join(missing_required)} are not in the column mapping dictionary. Provide these keys and retry."
            )

        # Validate optional columns if dashboard generation is enabled
        if self.generate_dashboard:
            missing_optional = optional_cols - self.col_mapping.keys()
            if missing_optional:
                raise InvalidColumnMappingException(
                    f"{', '.join(missing_optional)} are required when 'generate_dashboard' is enabled. Provide these keys and retry."
                )
        # Validate ID column if multiple devices is enabled
        if self.multiple_devices:
            if "id" not in self.col_mapping.keys():
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
                iat_method=self.metrics_config.get("iat_method"),
                frequency=self.metrics_config.get("frequency"),
                accuracy=AccuracyConfig(**self.metrics_config.get("accuracy")),
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
            df = pl.read_csv(self.file_path)
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
            df = df.sort(by=[self.col_mapping["id"], self.col_mapping["date"]])
        else:
            df = df.sort(by=self.col_mapping["date"])

        df_metrics = None
        # based on the selected dimensions, instantiate the classes.
        if Dimension.VALIDITY.value in self.dimensions:
            logger.info("Computing validity metric...")

            df_metrics = Validity(
                df, self.col_mapping, self.multiple_devices
            ).compute_metric()

            logger.info("Validity metric completed...")
            logger.info(f"First few rows: {df_metrics.head()}")

        if Dimension.ACCURACY.value in self.dimensions:
            logger.info("Computing accuracy metric...")
            df_metrics = Accuracy(
                df,
                self.col_mapping,
                self.metrics_config.accuracy,
                self.multiple_devices,
            ).compute_metric()
            logger.info("Accuracy metric completed...")

        # base metric- compute_metric_many(df)
        # return json of results i.e total_invalid, total_inaccurate, validity_record.
        # compute_metric(df)
        # generate
        # log

    def __repr__(self):
        print("<DataQualityScore>")


# compute_score - uses equal by default
# compute score_ahp - uses ahp alone (requires ahp config)
# compute score_ensemble(requires_ahp config)

df_with_metrics = DataQualityScore(
    "./Abyei_water_meters.csv",
    multiple_devices=True,
    dimensions=[Dimension.VALIDITY.value, Dimension.ACCURACY.value],
    col_mapping={
        "latitude": "OGI_LAT",
        "longitude": "OGI_LONG",
        "date": "TAG_VALUE_DATE",
        "value": "TAG_VALUE_RAW",
        "id": "DEVICE_ID",
    },
    metrics_config={
        # default - will use min IAT to compute the expected frequency.
        "frequency": "1d",
        "accuracy": {
            "ensemble": False,
            "algorithms": [
                OutlierDetectionAlgorithm.IF.value,
                # OutlierDetectionAlgorithm.IQR.value,
            ],
        },
    },
).compute_metrics()


# report - generate report
# summary
# dashboard
# json
# heatmap - completeness, accuracy, timeliness, consistency
# if you don't want completeness, disable it. it's default, it looks for null.
# baesd on th eminimum IAT, it computes the
