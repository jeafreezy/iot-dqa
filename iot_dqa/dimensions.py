import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from iot_dqa.utils.logger import logger
from iot_dqa.utils.configs import MetricsConfig, timer
from iot_dqa.utils.enums import (
    ColumnMappingColumnName,
    CompletenessStrategy,
    ColumnName,
    FrequencyCalculationMethod,
    OutlierDetectionAlgorithm,
)
import optuna


class BaseMetric:
    def __init__(
        self,
        df: pl.DataFrame,
        col_mapping: dict[str, str],
        multiple_devices: bool = False,
        config: MetricsConfig = None,
    ):
        self.df = df
        self.col_mapping = col_mapping
        self.multiple_devices = multiple_devices
        self.config = config

    def compute_metric(self):
        """
        Calculate the metric for the dimension.
        """
        ...


class Validity(BaseMetric):
    @timer
    def compute_validity(self) -> pl.DataFrame:
        """
        Computes the validity metric for the dataset.

        Returns:
            pl.DataFrame: A DataFrame with an additional column indicating validity (1 for valid, 0 for invalid).
        """
        if self.multiple_devices:
            validity_df = self.df.with_columns(
                pl.when(
                    (pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value]) == 0)
                    | (
                        pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value])
                        .diff()
                        .fill_null(1)
                        < 0
                    ).over(self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value])
                )
                .then(0)
                .otherwise(1)
                .alias(ColumnName.VALIDITY.value)
            )
        else:
            validity_df = self.df.with_columns(
                pl.when(
                    (pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value]) == 0)
                    | (
                        pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value])
                        .diff()
                        .fill_null(0)
                        < 0
                    )
                )
                .then(0)
                .otherwise(1)
                .alias(ColumnName.VALIDITY.value)
            )
        logger.info(
            f"Validity metric computed successfully: Basic statistics -> {validity_df[ColumnName.VALIDITY.value].value_counts()}",
        )
        return validity_df

    def compute_metric(self) -> pl.DataFrame:
        """
        Computes the validity metric for each record.
        This method evaluates the validity of a cumulative timeseries dataset, which is expected to have positive values.
        It identifies periods where the data either drops to zero or exhibits a negative difference between consecutive observations.
        """
        return self.compute_validity()

    def compute_score(self) -> pl.DataFrame:
        """
        Computes the validity score for each records.
        """
        return self.compute_metric()


class Accuracy(BaseMetric):
    @timer
    def median_absolute_deviation(self, df_with_outliers: pl.DataFrame) -> pl.DataFrame:
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
        accuracy_config = self.config.accuracy

        mad_outliers = df_with_outliers if df_with_outliers is not None else self.df
        median = self.df[self.col_mapping[ColumnMappingColumnName.VALUE.value]].median()
        mad = (
            (self.df[self.col_mapping[ColumnMappingColumnName.VALUE.value]] - median)
            .abs()
            .median()
        )

        if self.multiple_devices:
            group_results = []
            for device_id, group in mad_outliers.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ):
                logger.info(f"Detecting MAD outliers for Device: **{device_id[0]}**")
                median = group[
                    self.col_mapping[ColumnMappingColumnName.VALUE.value]
                ].median()
                mad = (
                    (
                        group[self.col_mapping[ColumnMappingColumnName.VALUE.value]]
                        - median
                    )
                    .abs()
                    .median()
                )
                modified_z_score = (
                    0.6745
                    * (
                        group[self.col_mapping[ColumnMappingColumnName.VALUE.value]]
                        - median
                    )
                    / mad
                )
                outliers = (
                    modified_z_score.abs() > accuracy_config.mad_threshold
                ).cast(pl.Int8)
                group_df = group.with_columns(
                    pl.Series(ColumnName.MAD_OUTLIERS.value, outliers)
                )
                group_results.append(group_df)
            mad_outliers = pl.concat(group_results)
        else:
            modified_z_score = (
                0.6745
                * (
                    mad_outliers[self.col_mapping[ColumnMappingColumnName.VALUE.value]]
                    - median
                )
                / mad
            )
            outliers = (modified_z_score.abs() > accuracy_config.mad_threshold).cast(
                pl.Int8
            )
            mad_outliers = mad_outliers.with_columns(
                pl.Series(ColumnName.MAD_OUTLIERS.value, outliers)
            )

        logger.info(
            f"MAD outliers detected successfully. Basic summary: {mad_outliers[ColumnName.MAD_OUTLIERS.value].value_counts()}"
        )
        return mad_outliers

    @timer
    def isolation_forest(self, df_with_outliers: pl.DataFrame) -> pl.DataFrame:
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

        df_with_IF_outliers = (
            df_with_outliers if df_with_outliers is not None else self.df
        )
        logger.info("Instantiating Isolation Forest")

        iso = IsolationForest(**self.config.accuracy.isolation_forest)

        if self.multiple_devices:
            group_results = []
            for device_id, group in df_with_IF_outliers.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ):
                logger.info(
                    f"Detecting Isolation Forest outliers for Device: **{device_id[0]}**"
                )
                outliers = iso.fit_predict(
                    group[self.col_mapping[ColumnMappingColumnName.VALUE.value]]
                    .to_numpy()
                    .reshape(-1, 1)
                )
                outliers = np.where(outliers == -1, 1, 0)
                group_df = group.with_columns(
                    pl.Series(ColumnName.IF_OUTLIERS.value, outliers)
                )
                group_results.append(group_df)

            df_with_IF_outliers = pl.concat(group_results)

        else:
            outliers = iso.fit_predict(
                df_with_IF_outliers[
                    self.col_mapping[ColumnMappingColumnName.VALUE.value]
                ]
                .to_numpy()
                .reshape(-1, 1)
            )
            outliers = np.where(outliers == -1, 1, 0)
            df_with_IF_outliers = df_with_IF_outliers.with_columns(
                pl.Series(ColumnName.IF_OUTLIERS.value, outliers)
            )
        logger.info(
            f"Isolation Forest outliers detected successfully. Basic summary: {df_with_IF_outliers[ColumnName.IF_OUTLIERS.value].value_counts()}"
        )
        return df_with_IF_outliers

    @timer
    def inter_quartile_range(self, df_with_outliers: pl.DataFrame) -> pl.DataFrame:
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
        accuracy_config = self.config.accuracy
        iqr_outliers = df_with_outliers if df_with_outliers is not None else self.df

        def objective(trial, device_df):
            q1 = trial.suggest_float(
                "q1",
                accuracy_config.iqr_optuna_q1_min,
                accuracy_config.iqr_optuna_q1_max,
            )
            q3 = trial.suggest_float(
                "q3",
                accuracy_config.iqr_optuna_q3_min,
                accuracy_config.iqr_optuna_q3_max,
            )
            iqr = q3 - q1
            lower_bound = (
                device_df[
                    self.col_mapping[ColumnMappingColumnName.VALUE.value]
                ].quantile(q1)
                - 1.5 * iqr
            )
            upper_bound = (
                device_df[
                    self.col_mapping[ColumnMappingColumnName.VALUE.value]
                ].quantile(q3)
                + 1.5 * iqr
            )

            outliers = device_df.with_columns(
                pl.when(
                    (
                        pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value])
                        < lower_bound
                    )
                    | (
                        pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value])
                        > upper_bound
                    )
                )
                .then(1)
                .otherwise(0)
                .alias(ColumnName.IQR_OUTLIERS.value)
            )
            return outliers[ColumnName.IQR_OUTLIERS.value].sum()

        if self.multiple_devices:
            group_results = []
            for device_id, group in iqr_outliers.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ):
                logger.info(f"Detecting IQR outliers for Device: **{device_id[0]}**")
                if accuracy_config.optimize_iqr_with_optuna:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(
                        lambda trial: objective(trial, group),
                        n_trials=accuracy_config.iqr_optuna_trials,
                    )

                    best_q1 = study.best_params["q1"]
                    best_q3 = study.best_params["q3"]

                iqr = best_q3 - best_q1
                lower_bound = (
                    group[
                        self.col_mapping[ColumnMappingColumnName.VALUE.value]
                    ].quantile(best_q1)
                    - 1.5 * iqr
                )
                upper_bound = (
                    group[
                        self.col_mapping[ColumnMappingColumnName.VALUE.value]
                    ].quantile(best_q3)
                    + 1.5 * iqr
                )

                group_df = group.with_columns(
                    pl.when(
                        (
                            pl.col(
                                self.col_mapping[ColumnMappingColumnName.VALUE.value]
                            )
                            < lower_bound
                        )
                        | (
                            pl.col(
                                self.col_mapping[ColumnMappingColumnName.VALUE.value]
                            )
                            > upper_bound
                        )
                    )
                    .then(1)
                    .otherwise(0)
                    .alias(ColumnName.IQR_OUTLIERS.value)
                )
                group_results.append(group_df)

            iqr_outliers = pl.concat(group_results)
        else:
            if accuracy_config.optimize_iqr_with_optuna:
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: objective(trial, iqr_outliers),
                    n_trials=accuracy_config.iqr_optuna_trials,
                )

                best_q1 = study.best_params["q1"]
                best_q3 = study.best_params["q3"]

            iqr = best_q3 - best_q1
            lower_bound = (
                iqr_outliers[
                    self.col_mapping[ColumnMappingColumnName.VALUE.value]
                ].quantile(best_q1)
                - 1.5 * iqr
            )
            upper_bound = (
                iqr_outliers[
                    self.col_mapping[ColumnMappingColumnName.VALUE.value]
                ].quantile(best_q3)
                + 1.5 * iqr
            )

            iqr_outliers = iqr_outliers.with_columns(
                pl.when(
                    (
                        pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value])
                        < lower_bound
                    )
                    | (
                        pl.col(self.col_mapping[ColumnMappingColumnName.VALUE.value])
                        > upper_bound
                    )
                )
                .then(1)
                .otherwise(0)
                .alias(ColumnName.IQR_OUTLIERS.value)
            )

        logger.info(
            f"IQR outliers detected successfully. Basic summary: {iqr_outliers[ColumnName.IQR_OUTLIERS.value].value_counts()}"
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

        df_with_outliers = self.df
        accuracy_config = self.config.accuracy
        if OutlierDetectionAlgorithm.IF.value in accuracy_config.algorithms:
            df_with_outliers = self.isolation_forest(df_with_outliers)
        if OutlierDetectionAlgorithm.IQR.value in accuracy_config.algorithms:
            df_with_outliers = self.inter_quartile_range(df_with_outliers)
        if OutlierDetectionAlgorithm.MAD.value in accuracy_config.algorithms:
            df_with_outliers = self.median_absolute_deviation(df_with_outliers)

        outlier_columns = [
            ColumnName.MAD_OUTLIERS.value,
            ColumnName.IF_OUTLIERS.value,
            ColumnName.IQR_OUTLIERS.value,
        ]
        relevant_columns = [
            col for col in outlier_columns if col in df_with_outliers.columns
        ]

        if relevant_columns:
            # Combine outlier columns by taking the maximum value across them
            df_with_outliers = df_with_outliers.with_columns(
                pl.max_horizontal([pl.col(col) for col in relevant_columns])
                .cast(pl.Int8)
                .alias(ColumnName.ACCURACY.value)
            )
        # remove outlier columns
        for col in outlier_columns:
            if col in df_with_outliers.columns:
                df_with_outliers = df_with_outliers.drop(col)

        return df_with_outliers


class Completeness(BaseMetric):
    @timer
    def compute_completeness_metrics(self) -> pl.DataFrame:
        """
        Compute the completeness metrics using different strategy.
        """
        completeness_df = self.df

        if self.config.completeness_strategy == CompletenessStrategy.ONLY_NULLS.value:
            completeness_df = completeness_df.with_columns(
                pl.when(
                    pl.col(
                        self.col_mapping[ColumnMappingColumnName.VALUE.value]
                    ).is_null()
                )
                .then(0)
                .otherwise(1)
                .alias(ColumnName.COMPLETENESS.value)
                .cast(pl.Int8)
            )
        elif self.config.completeness_strategy == CompletenessStrategy.ACCURACY.value:
            # if accuracy is not computed, instantiate and compute accuracy here.
            if ColumnName.ACCURACY.value not in completeness_df.columns:
                completeness_df = Accuracy(
                    completeness_df,
                    self.col_mapping,
                    self.multiple_devices,
                    self.config,
                ).compute_metric()

                completeness_df = completeness_df.with_columns(
                    pl.when(pl.col(ColumnName.ACCURACY.value) == 0)
                    .then(0)
                    .otherwise(1)
                    .alias(ColumnName.COMPLETENESS.value)
                    .cast(pl.Int8)
                )
                # remove accuracy column
                completeness_df = completeness_df.drop(ColumnName.ACCURACY.value)

            else:
                completeness_df = completeness_df.with_columns(
                    pl.when(pl.col(ColumnName.ACCURACY.value) == 0)
                    .then(0)
                    .otherwise(1)
                    .alias(ColumnName.COMPLETENESS.value)
                    .cast(pl.Int8)
                )

        return completeness_df

    def compute_metric(self) -> pl.DataFrame:
        """
        Computes the completeness metric for the devices.
        """
        completeness_df = self.compute_completeness_metrics()
        logger.info(
            f"Completeness metric computed successfully: Basic statistics -> {(completeness_df[ColumnName.COMPLETENESS.value].value_counts(),)}",
        )
        return completeness_df


class Timeliness(BaseMetric):
    def compute_metric(
        self,
    ) -> pl.DataFrame:
        """
        Computes the Inter-Arrival Time Regularity metric for timeliness assessment.

        Returns:
            pl.DataFrame: A dataframe containing the timeliness metric per record.
        """

        # Calculate Inter-Arrival Time (IAT)
        df = self.df.with_columns(
            (
                pl.col(self.col_mapping[ColumnMappingColumnName.DATE.value])
                .cast(pl.Datetime)  # Ensure the column is in datetime format
                .diff()
                .over(self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value])
                .alias(ColumnName.IAT.value)
                / 1_000_000  # Convert duration to seconds
            )
        )

        # Fill null values with the min or mode based on the calculation method, per device
        if self.config.timeliness.iat_method == FrequencyCalculationMethod.MIN.value:
            fill_values = df.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ).agg(pl.col(ColumnName.IAT.value).min().alias(ColumnName.FILL_VALUE.value))
        elif self.config.timeliness.iat_method == FrequencyCalculationMethod.MODE.value:
            fill_values = df.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ).agg(
                pl.col(ColumnName.IAT.value).mode().alias(ColumnName.FILL_VALUE.value)
            )
        else:
            fill_values = df.select(
                pl.lit(0).alias(ColumnName.FILL_VALUE.value)
            )  # Default fallback for all devices

        # Join the fill values back to the original dataframe
        df = df.join(
            fill_values,
            on=self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value],
            how="left",
        )

        # Fill null values in IAT column with the per-device fill value
        df = df.with_columns(
            pl.col(ColumnName.IAT.value).fill_null(pl.col(ColumnName.FILL_VALUE.value))
        )

        # Deduce the expected interval based on the iat_method in the configuration
        if self.config.timeliness.iat_method == FrequencyCalculationMethod.MIN.value:
            expected_interval = df.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ).agg(
                pl.col(ColumnName.IAT.value)
                .min()
                .alias(
                    ColumnName.EXPECTED_INTERVAL.value,
                )
            )
        elif self.config.timeliness.iat_method == FrequencyCalculationMethod.MODE.value:
            expected_interval = df.group_by(
                self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value]
            ).agg(
                pl.col(ColumnName.IAT.value)
                .mode()
                .alias(
                    ColumnName.EXPECTED_INTERVAL.value,
                )
            )

        # Join the expected interval back to the original dataframe
        df = df.join(
            expected_interval,
            on=self.col_mapping[ColumnMappingColumnName.DEVICE_ID.value],
        )

        # Compute Relative Absolute Error (RAE) based on the deduced expected interval
        df = df.with_columns(
            pl.when(pl.col(ColumnName.EXPECTED_INTERVAL.value) != 0)
            .then(
                (
                    (
                        pl.col(ColumnName.IAT.value)
                        - pl.col(ColumnName.EXPECTED_INTERVAL.value)
                    ).abs()
                    / pl.col(ColumnName.EXPECTED_INTERVAL.value)
                )
            )
            .otherwise(0)
            .alias(ColumnName.RAE.value)
        )

        # Compute goodness and penalty scores
        df = df.with_columns(
            [
                pl.when(pl.col(ColumnName.RAE.value) <= 0.5)
                .then(1 - 2 * pl.col(ColumnName.RAE.value))
                .otherwise(0)
                .alias(ColumnName.GOODNESS.value),
                pl.when(pl.col(ColumnName.RAE.value) > 0.5)
                .then(2 * pl.col(ColumnName.RAE.value))
                .otherwise(0)
                .alias(ColumnName.PENALTY.value),
            ]
        )

        # Calculate the timeliness score per record
        df = df.with_columns(
            (pl.col(ColumnName.GOODNESS.value) / (1 + pl.col(ColumnName.PENALTY.value)))
            .cast(pl.Int8)
            .alias(ColumnName.TIMELINESS.value)
        )

        # Drop intermediate columns used for calculations
        df = df.drop(
            [
                ColumnName.GOODNESS.value,
                ColumnName.PENALTY.value,
                ColumnName.RAE.value,
                ColumnName.IAT.value,
                ColumnName.EXPECTED_INTERVAL.value,
                ColumnName.FILL_VALUE.value,
            ]
        )
        logger.info(
            f"Timeliness metric computed successfully: Basic statistics -> {(df[ColumnName.TIMELINESS.value].value_counts(),)}",
        )
        return df
