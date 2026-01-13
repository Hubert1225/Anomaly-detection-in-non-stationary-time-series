"""This module provides descriptors for determine basic time
series characteristics, e.g. length, values range, number of anomalies
"""

from typing import Any
import numpy as np
from sktime.transformations.series.summarize import SummaryTransformer

from base import TimeSeriesWithAnoms, TimeSeriesDescriptor


class BasicProperties(TimeSeriesDescriptor):

    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:

        features: dict[str, Any] = {}

        # length of time series
        features["length"] = ts.values.shape[0]

        # number of missing values
        features["no. missing"] = np.sum(np.isnan(ts.values))

        return features


class BasicStatistics(TimeSeriesDescriptor):

    def __init__(self, quantiles: tuple[float, ...] = (0.25, 0.75)):
        self.quantiles: tuple[float, ...] = quantiles

    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:

        summary_transformer = SummaryTransformer(
            ("mean", "median", "std", "min", "max"), quantiles=self.quantiles
        )
        return summary_transformer.fit_transform(ts.values).iloc[0].to_dict()


class AnomaliesStatistics(TimeSeriesDescriptor):

    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:

        features: dict[str, Any] = {}
        anoms_lens = ts.anoms_lens()

        # number of anomalies (of anomalous subsequences)
        features["no. anomalies"] = anoms_lens.shape[0]

        # contamination in % (sum_anoms_len * 100 / series_len)
        features["contamination"] = anoms_lens.sum() * 100 / ts.values.shape[0]

        # anomaly length
        anomaly_length = anoms_lens.iloc[0]
        if not (anoms_lens == anomaly_length).all():
            raise AssertionError
        features["anomaly length"] = anomaly_length

        return features
