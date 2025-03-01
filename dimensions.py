import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from logger import logger


class BaseMetric:
    def __init__(
        self,
        df: pl.DataFrame,
        col_mapping: dict[str, str],
        multiple_devices: bool = False,
    ):
        self.df = df
        self.col_mapping = col_mapping
        self.multiple_devices = multiple_devices

    def compute_metric(self):
        """
        Calculate the z-score of the input.
        """
        ...


class Validity(BaseMetric):
    # detect nulls, detect inaccurate, estimate expected days,
    # estimate sent values, return score
    def compute_metric(self):
        """
        Computes the validity metric for each records.
        Given that the timeseries is expected to be cummulative with positive values.
        Validity will check for periods where the data drops to 0 or the difference between the current and past observations are
        """
        validity_df = None
        if self.multiple_devices:
            validity_df = self.df.with_columns(
                pl.when(
                    (pl.col(self.col_mapping["value"]) == 0)
                    | (pl.col(self.col_mapping["value"]).diff().fill_null(1) < 0).over(
                        self.col_mapping["id"]
                    )
                )
                .then(0)  # 0 if invalid
                .otherwise(1)  # Otherwise 1
                .alias("validity")
            )
        else:
            validity_df = self.df.with_columns(
                pl.when(
                    (pl.col(self.col_mapping["value"]) == 0)
                    | (pl.col(self.col_mapping["value"]).diff().fill_null(0) < 0)
                )
                .then(0)  # 0 if invalid
                .otherwise(1)  # Otherwise 1
                .alias("validity")
            )
        logger.info(
            f"Validity metric computed successfully: Basic statistics -> {(validity_df['validity'].value_counts(),)}",
        )
        return validity_df

    def compute_score(self):
        # if multiple pass it to validity so it can use group by
        # also pass df and column mapping@
        # basic stats, invalid column, valid column etc.
        # date with most invalidity, date with most validity.
        # device with most invalidity
        # device with most validity.
        ...


class Accuracy(BaseMetric):
    # IQR method
    def detect_outliers_iqr(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (data < lower_bound) | (data > upper_bound)

    # MAD method
    def detect_outliers_mad(data, threshold=3):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_score = 0.6745 * (data - median) / mad
        return np.abs(modified_z_score) > threshold

    # Isolation Forest method
    def detect_outliers_isolation_forest(data):
        iso = IsolationForest(random_state=42, contamination="auto")
        return iso.fit_predict(data.reshape(-1, 1)) == -1

    def mean_absolute_deviation(y_true, y_pred):
        """
        Calculate the mean absolute error of the model.
        """
        ...

    def isolation_forest(df):
        """
        Detect anomalies in the data using Isolation Forest.
        """
        ...

    def inter_quartile_range(x):
        """
        Calculate the inter-quartile range of the input.
        """
        ...

    def ensemble(): ...

    def compute_metric(): ...


class Completeness(BaseMetric):
    # detect nulls, detect inaccurate, estimate expected days,
    # estimate sent values, return score
    def compute_metric(x):
        """
        Calculate the z-score of the input.
        """
        ...


class Timeliness(BaseMetric):
    # detect nulls, detect inaccurate, estimate expected days,
    # estimate sent values, return score
    def compute_metric(
        df: pl.DataFrame, device_col="DEVICE_ID", date_col="DATE", mode=1
    ):
        """
        Computes the Inter-Arrival Time Regularity metric for timeliness assessment using Polars.

        Parameters:
            df (pl.DataFrame): The input dataframe containing timestamps and device IDs.
            device_col (str): The name of the device ID column.
            date_col (str): The name of the date column.

        Returns:
            pl.DataFrame: A dataframe containing the timeliness metric per record.
        """

        # Compute inter-arrival time (IAT) in days for each device
        df = df.sort([device_col, date_col])
        df = df.with_columns(
            (pl.col(date_col).diff().dt.total_days().over(device_col)).alias("IAT")
        )

        # Fill NaN values (first row per device) with 1 (since mode is 1)
        df = df.fill_null(1)

        # Compute Relative Absolute Error (RAE) assuming mode = 1 day
        df = df.with_columns(
            pl.col("IAT").sub(1).abs().alias("RAE")  # Fixed absolute value computation
        )

        # Compute goodness and penalty
        df = df.with_columns(
            pl.when(pl.col("RAE") <= 0.5)
            .then(1 - 2 * pl.col("RAE"))
            .otherwise(0)
            .alias("goodness"),
            pl.when(pl.col("RAE") > 0.5)
            .then(2 * pl.col("RAE"))
            .otherwise(0)
            .alias("penalty"),
        )

        # Compute the timeliness score per record
        df = df.with_columns(
            (pl.col("goodness") / (1 + pl.col("penalty")))
            .cast(pl.Int8)
            .alias("TIMELINESS")
        )

        return df.select(
            [
                device_col,
                date_col,
                "IAT",
                "RAE",
                "TIMELINESS",
                "CUMMULATIVE_CONSUMPTION",
                "VALIDITY",
            ]
        )
