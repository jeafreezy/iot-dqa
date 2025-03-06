import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from logger import logger
from utils import timer, AccuracyConfig
from enums import OutlierDetectionAlgorithm
import optuna


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

    @timer
    def compute_validity(self) -> pl.DataFrame:
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

    def compute_metric(self) -> pl.DataFrame:
        """
        Computes the validity metric for each records.
        Given that the timeseries is expected to be cummulative with positive values.
        Validity will check for periods where the data drops to 0 or the difference between the current and past observations are
        """
        return self.compute_validity()

    def compute_score(self) -> pl.DataFrame:
        """
        Computes the validity score for each records.
        """
        return self.compute_metric()


class Accuracy(BaseMetric):
    def __init__(self, df, col_mapping, config=AccuracyConfig, multiple_devices=False):
        super().__init__(df, col_mapping, multiple_devices)
        self.config = config

    @timer
    def median_absolute_deviation(self) -> pl.DataFrame:
        """
        Calculate the Median Absolute Deviation (MAD) for outlier detection.
        This method computes the MAD for the specified column in the DataFrame.
        If multiple devices are present, it calculates the MAD for each device
        separately and concatenates the results. The MAD outliers are identified
        based on a modified Z-score and using optuna specified threshold.
        Returns:
            pl.DataFrame: A DataFrame with an additional column "MAD_outliers"
            indicating the presence of outliers (1 for outlier, 0 for non-outlier).
        """

        mad_outliers = None
        median = self.df[self.col_mapping["value"]].median()
        mad = (self.df[self.col_mapping["value"]] - median).abs().median()

        if self.multiple_devices:
            group_results = []
            for device_id, group in self.df.group_by(self.col_mapping["id"]):
                logger.info(f"Detecting MAD outliers for Device: **{device_id[0]}**")
                median = group[self.col_mapping["value"]].median()
                mad = (group[self.col_mapping["value"]] - median).abs().median()
                modified_z_score = (
                    0.6745 * (group[self.col_mapping["value"]] - median) / mad
                )
                outliers = (modified_z_score.abs() > self.config.mad_threshold).cast(
                    pl.Int8
                )
                group_df = group.with_columns(pl.Series("MAD_outliers", outliers))
                group_results.append(group_df)
            mad_outliers = pl.concat(group_results)
        else:
            modified_z_score = (
                0.6745 * (self.df[self.col_mapping["value"]] - median) / mad
            )
            outliers = (modified_z_score.abs() > self.config.mad_threshold).cast(
                pl.Int8
            )
            mad_outliers = self.df.with_columns(pl.Series("MAD_outliers", outliers))

        logger.info(
            f"MAD outliers detected successfully. Basic summary: {mad_outliers['MAD_outliers'].value_counts()}"
        )
        return mad_outliers

    @timer
    def isolation_forest(self):
        """
        Detects outliers in the dataset using the Isolation Forest algorithm.
        This method applies the Isolation Forest algorithm to detect outliers in the dataset.
        If the dataset contains multiple devices, it processes each device's data separately
        and concatenates the results. Otherwise, it processes the entire dataset at once.
        Returns:
            pl.DataFrame: A DataFrame with an additional column "IF_outliers" indicating
                          the presence of outliers (1 if outlier, 0 if not).
        Raises:
            ValueError: If the dataset or column mappings are not properly configured.
        Notes:
            - The Isolation Forest is instantiated with a random state of 42 and auto contamination.
            - The method logs the progress and results of the outlier detection process.
        """

        df_with_IF_outliers = None
        logger.info("Instantiating Isolation Forest")
        # todo - configure IF
        iso = IsolationForest(random_state=42, contamination="auto")

        if self.multiple_devices:
            group_results = []
            for device_id, group in self.df.group_by(self.col_mapping["id"]):
                logger.info(
                    f"Detecting Isolation Forest outliers for Device: **{device_id[0]}**"
                )
                outliers = iso.fit_predict(
                    group[self.col_mapping["value"]].to_numpy().reshape(-1, 1)
                )
                outliers = np.where(outliers == -1, 1, 0)
                group_df = group.with_columns(pl.Series("IF_outliers", outliers))
                group_results.append(group_df)
            df_with_IF_outliers = pl.concat(group_results)

        else:
            outliers = iso.fit_predict(
                self.df[self.col_mapping["value"]].to_numpy().reshape(-1, 1)
            )
            outliers = np.where(outliers == -1, 1, 0)
            df_with_IF_outliers = self.df.with_columns(
                pl.Series("IF_outliers", outliers)
            )
        logger.info(
            f"Isolation Forest outliers detected successfully. Basic summary: {df_with_IF_outliers['IF_outliers'].value_counts()}"
        )
        return df_with_IF_outliers

    @timer
    def inter_quartile_range(self) -> pl.DataFrame:
        """
        Detects outliers in the dataset using the Inter-Quartile Range (IQR) method with the help of Optuna for
        hyperparameter optimization.
        This method can handle multiple devices by grouping the data based on device IDs and applying the IQR
        outlier detection for each group separately. It uses Optuna to find the optimal lower and upper quantile
        bounds instead of fixed cutoffs.
        Returns:
            pl.DataFrame: A DataFrame with an additional column "IQR_outliers" indicating outliers (1 for outlier,
            0 for non-outlier).
        """
        # defaults for IQR

        best_q1 = 0.25
        best_q3 = 0.75

        iqr_outliers = None

        def objective(trial, device_df):
            q1 = trial.suggest_float(
                "q1", self.config.iqr_optuna_q1_min, self.config.iqr_optuna_q1_max
            )
            q3 = trial.suggest_float(
                "q3", self.config.iqr_optuna_q3_min, self.config.iqr_optuna_q3_max
            )
            iqr = q3 - q1
            lower_bound = device_df[self.col_mapping["value"]].quantile(q1) - 1.5 * iqr
            upper_bound = device_df[self.col_mapping["value"]].quantile(q3) + 1.5 * iqr

            outliers = device_df.with_columns(
                pl.when(
                    (pl.col(self.col_mapping["value"]) < lower_bound)
                    | (pl.col(self.col_mapping["value"]) > upper_bound)
                )
                .then(1)
                .otherwise(0)
                .alias("IQR_outliers")
            )
            return outliers["IQR_outliers"].sum()

        if self.multiple_devices:
            group_results = []
            for device_id, group in self.df.group_by(self.col_mapping["id"]):
                logger.info(f"Detecting IQR outliers for Device: **{device_id[0]}**")
                if self.config.optimize_iqr_with_optuna:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(
                        lambda trial: objective(trial, group),
                        n_trials=self.config.iqr_optuna_trials,
                    )

                    best_q1 = study.best_params["q1"]
                    best_q3 = study.best_params["q3"]

                iqr = best_q3 - best_q1
                lower_bound = (
                    group[self.col_mapping["value"]].quantile(best_q1) - 1.5 * iqr
                )
                upper_bound = (
                    group[self.col_mapping["value"]].quantile(best_q3) + 1.5 * iqr
                )

                group_df = group.with_columns(
                    pl.when(
                        (pl.col(self.col_mapping["value"]) < lower_bound)
                        | (pl.col(self.col_mapping["value"]) > upper_bound)
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("IQR_outliers")
                )
                group_results.append(group_df)
            iqr_outliers = pl.concat(group_results)
        else:
            if self.config.optimize_iqr_with_optuna:
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: objective(trial, self.df),
                    n_trials=self.config.iqr_optuna_trials,
                )

                best_q1 = study.best_params["q1"]
                best_q3 = study.best_params["q3"]

            iqr = best_q3 - best_q1
            lower_bound = (
                self.df[self.col_mapping["value"]].quantile(best_q1) - 1.5 * iqr
            )
            upper_bound = (
                self.df[self.col_mapping["value"]].quantile(best_q3) + 1.5 * iqr
            )

            iqr_outliers = self.df.with_columns(
                pl.when(
                    (pl.col(self.col_mapping["value"]) < lower_bound)
                    | (pl.col(self.col_mapping["value"]) > upper_bound)
                )
                .then(1)
                .otherwise(0)
                .alias("IQR_outliers")
            )

        logger.info(
            f"IQR outliers detected successfully. Basic summary: {iqr_outliers['IQR_outliers'].value_counts()}"
        )
        return iqr_outliers

    def compute_metric(self) -> pl.DataFrame:
        """
        Computes the metric by detecting outliers using specified algorithms.
        This method checks the configuration for the specified outlier detection
        algorithms and applies them to the data. The supported algorithms are:
        Isolation Forest (IF), Inter-Quartile Range (IQR), and Median Absolute
        Deviation (MAD). The method returns a DataFrame with the detected outliers.
        Returns:
            pl.DataFrame: A DataFrame containing the data with detected outliers.
        """

        df_with_outliers = None
        if OutlierDetectionAlgorithm.IF.value in self.config.algorithms:
            df_with_outliers = self.isolation_forest()
        if OutlierDetectionAlgorithm.IQR.value in self.config.algorithms:
            df_with_outliers = self.inter_quartile_range()
        if OutlierDetectionAlgorithm.MAD.value in self.config.algorithms:
            df_with_outliers = self.median_absolute_deviation()
        return df_with_outliers

    def compute_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes the score for the given DataFrame based on the presence of outlier columns.
        Args:
            df (pl.DataFrame): The input DataFrame containing the data to be scored.
        Returns:
            pl.DataFrame: The DataFrame with an additional 'accuracy' column if outlier columns are present.
        Notes:
            - The method checks for the presence of 'MAD_outliers', 'IF_outliers', and 'IQR_outliers' columns.
            - If any of these columns are present, it creates a new 'accuracy' column by combining the values of these columns.
            - If no outlier columns are present, the original DataFrame is returned without modification.
        """

        if self.config.ensemble:
            columns_to_check = ["MAD_outliers", "IF_outliers", "IQR_outliers"]
            existing_columns = [col for col in columns_to_check if col in df.columns]
            if existing_columns:
                df_with_outliers = df.with_columns(
                    pl.all(existing_columns).all().alias("accuracy")
                )
                return df_with_outliers
        return df_with_outliers


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


# added checks for ensemble
# implemented accuracy/isolation forest
