import os
from typing import Callable
import pandas as pd

from base import TimeSeriesWithAnoms, SeriesSet
from data_loading import load_series_set
from transform.nonstat_removal import (
    Differencing,
    Detrending,
    RollingMeanSmoother,
    RobustDifferencing,
    BoxCox,
)
from transform.nonstat_addition import TrendAdder
from results_utils import load_series_descriptions, add_seasonality_column
from params import params

SERIES_SETS_CATALOGUE = os.path.join("data", "series_sets")

sets_nonstat_removal = params["transform_series"]["sets_nonstat_removal"]
sets_add_trend = params["transform_series"]["sets_add_trend"]


def get_detrend_window_len_getter(
    series_set_desc: pd.DataFrame, default_window_len: int = 400
) -> Callable[[TimeSeriesWithAnoms], int]:
    """Returns function that provides detrending window length
    for a TimeSeriesWithAnoms object

    The function works as follows: is `ClearSeasonality` for the series is
    True, return 2 * season_len. Otherwise, return some default value
    """
    series_set_desc = add_seasonality_column(series_set_desc)
    series_set_desc.index = series_set_desc.index.astype("str")
    window_lengths = series_set_desc["Season length"].copy() * 2
    window_lengths[~series_set_desc["ClearSeasonality"]] = default_window_len
    return lambda x: int(window_lengths.loc[x.name])


def get_season_len_getter(
    series_set_desc: pd.DataFrame,
) -> Callable[[TimeSeriesWithAnoms], int]:
    series_set_desc.index = series_set_desc.index.astype("str")
    season_lens = series_set_desc["Season length"]
    return lambda x: int(season_lens.loc[x.name])


def check_transformed_series_set(
    original_set: SeriesSet,
    transformed_set: SeriesSet,
) -> bool:
    """Returns True if correct,
    False if inconsistencies are detected
    """
    original_set_dict = original_set.get_series_dict()
    for transformed_ts in transformed_set.series_set:
        original_ts = original_set_dict[transformed_ts.name]
        # check if number of anomalies is the same
        if original_ts.anoms.shape[0] != transformed_ts.anoms.shape[0]:
            return False
    return True


def save_series_set(series_set: SeriesSet) -> None:
    set_path = os.path.join(SERIES_SETS_CATALOGUE, series_set.name)
    os.makedirs(set_path, exist_ok=False)
    series_set.save(set_path)


if __name__ == "__main__":

    # TRANSFORMATIONS REMOVING NONSTATIONARITY

    print(f"Nonstationary removing transformations ...")
    for set_name in sets_nonstat_removal:

        # load series set and its description
        series_set = load_series_set(set_name)
        series_set_desc = load_series_descriptions(set_name)

        # create list of transformations
        transforms = [
            RollingMeanSmoother(
                window_len=params["transform_series"]["rolling_smoother_window"]
            ),
            Detrending(
                window_len_getter=get_detrend_window_len_getter(series_set_desc)
            ),
            Differencing(lag=1),
            Differencing(lag=get_season_len_getter(series_set_desc)),
            RobustDifferencing(
                lag=get_season_len_getter(series_set_desc),
                k=params["transform_series"]["robust_differencing_k"],
            ),
            BoxCox(),
        ]

        # create new series set using each transformation and save it
        print(set_name)
        for transform in transforms:
            series_set_transformed = transform.transform(series_set)
            if not check_transformed_series_set(
                original_set=series_set,
                transformed_set=series_set_transformed,
            ):
                raise RuntimeError("Transform produced inconsistent output!")
            save_series_set(series_set_transformed)

    # ADDING TREND

    print("Trend adding ...")
    for set_name in sets_add_trend:

        # load series set and its description
        series_set = load_series_set(set_name)

        # create series set with trend added
        trend_adder = TrendAdder()
        series_set_transformed = trend_adder.transform(series_set)
        if not check_transformed_series_set(
            original_set=series_set, transformed_set=series_set_transformed
        ):
            raise RuntimeError("Transform produced inconsistent output!")

        # save series set with added trend
        save_series_set(series_set_transformed)
