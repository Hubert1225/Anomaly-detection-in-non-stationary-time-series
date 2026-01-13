"""This module provides transformations that add some
nonstationary component to series values
"""

import numpy as np

from base import MapTransformation, TimeSeriesWithAnoms


class TrendAdder(MapTransformation):
    """Adds a random deterministic trend component to series
    values

    The random trend creation procedure is as follows:
    - choose randomly `scale_coef`, `scale_coef` ~ N(0, 10)
    - get range of values: `values_range = max(values) - min(values)`
    - get trend values: `np.linspace(0, values_range * scale_coef, num=series_len)`
    - add trend to series values

    """

    def __init__(self):
        super().__init__(name="addtrend")

    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:
        values_range = np.max(ts.values) - np.min(ts.values)
        trend = self.__get_random_trend(values_range, ts.values.shape[0])
        values_transformed = ts.values + trend
        return TimeSeriesWithAnoms(
            values=values_transformed, anoms=ts.anoms, name=ts.name
        )

    @staticmethod
    def __get_random_trend(values_range: float, series_len: int):
        scale_coef = np.random.normal(0, 10)
        return np.linspace(0, values_range * scale_coef, num=series_len)
