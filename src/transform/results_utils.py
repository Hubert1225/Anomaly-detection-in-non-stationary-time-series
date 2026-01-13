"""This module provides utilities for working
with results of the pipeline code - loading,
additional information extraction
"""

import os
from typing import Callable, Any
import pandas as pd

from base import TimeSeriesWithAnoms, SeriesSetResult, DetectionResultForSeries
from data_loading import (
    SET_SHORT_NAME_TO_NAME,
    load_series_set,
    load_transformed_series_set,
)


def load_series_descriptions(
    set_name: str, descriptions_dir: str = os.path.join("results", "describe_series")
) -> pd.DataFrame:
    """Loads time series metadata obtained in describe_series
    step for a given series set

    Args:
        set_name: short name of a series set
        descriptions_dir: path to the directory with descriptions
            files

    Returns:
        pd.DataFrame: dataframe with series describing features
            loaded from file
    """
    series_set_fullname = SET_SHORT_NAME_TO_NAME[set_name]
    return pd.read_csv(
        os.path.join(descriptions_dir, f"desc_{series_set_fullname}.csv"), index_col=0
    )


def add_trend_column(desc_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a new dataframe with added boolean `Trend` column

    The `Trend` is True for a series iff the ADF stationarity test
    result is that the series is not stationary but the KPSS test
    result is that the series is trend stationary, i.e. the series
    is stationary around a deterministic trend
    """
    trend = (~desc_df["Stationary"]) & desc_df["Trend stationary"]
    return desc_df.assign(Trend=trend)


def add_seasonality_column(
    desc_df: pd.DataFrame, autocorr_threshold: float = 0.4
) -> pd.DataFrame:
    """Returns a new dataframe with added boolean `ClearSeasonality`
    column

    For a series, `ClearSeasonality` is True iff its seasonal
    correlation is greater than `autocorr_threshold`
    """
    return desc_df.assign(
        ClearSeasonality=(desc_df["Season autocorrelation"] > autocorr_threshold)
    )


def get_results_df(
    result_set_dir: str,
    fields_funs: dict[
        str, Callable[[TimeSeriesWithAnoms, DetectionResultForSeries], Any]
    ],
) -> dict[str, pd.DataFrame | str]:
    """Reads set of detection results from given directory,
    creates DataFrame with metrics per series and provides
    it along with results metadata

    Args:
        result_set_dir: path to the directory from to load the SeriesSetResult object from
        fields_funs: dict of keys-values: {field_name -> callable}
            callable must take exactly two arguments: time series object and object of detection
            results for the series;
            for each time series, the result of calling the callable on the series and its detection
            result will be placed in column with name field_name

    Returns:
        dict: {
        'original_series_set': name of the original series set, one of:
            'mitbih_arrhythmia', 'mitbih_supra', 'ucr', 'srw'
        'transform': name of the transformation applied, one of:
            'rolling_mean', 'detrend', 'simple_diff', 'season_diff', 'box_cox',
            'season_diff_robust', 'add_trend', None (if none of above has been applied)
        'method': name of anomaly detection method,
        'results_df': pandas DataFrame, which index are series names, columns
            are field_name values from fields_funs, and values are results of fields_funs callables
    }
    """

    # load results object
    results_set = SeriesSetResult.load(result_set_dir)

    # get original set name (short one and long one)
    if "MITBIH_Arrhythmia" in results_set.set_name:
        set_long_name = "MITBIH_Arrhythmia"
        orig_set_name = "mitbih_arrhythmia"
    elif "MITBIH_Supraventricular" in results_set.set_name:
        set_long_name = "MITBIH_Supraventricular"
        orig_set_name = "mitbih_supra"
    elif "UCR" in results_set.set_name:
        set_long_name = "UCR"
        orig_set_name = "ucr"
    elif "SRW" in results_set.set_name:
        set_long_name = "SRW"
        orig_set_name = "srw"
    else:
        raise RuntimeError(
            f"Unknown original set name. Set name in results: {results_set.set_name}"
        )

    # get transform name
    if results_set.set_name == set_long_name:
        transform_name = None
    elif "_rollingmean" in results_set.set_name:
        transform_name = "rolling_mean"
    elif "_detrending" in results_set.set_name:
        transform_name = "detrend"
    elif results_set.set_name.endswith("_differencing_1"):
        transform_name = "simple_diff"
    elif "_robustdifferencing_dynamic" in results_set.set_name:
        transform_name = "season_diff_robust"
    elif "_differencing_dynamic" in results_set.set_name:
        transform_name = "season_diff"
    elif "_boxcox" in results_set.set_name:
        transform_name = "box_cox"
    elif "_addtrend" in results_set.set_name:
        transform_name = "add_trend"
    else:
        raise RuntimeError(
            f"Unknown series set transform. Set name in results: {results_set.set_name}"
        )

    # get series set
    if transform_name is None:
        series_set = load_series_set(orig_set_name)
    else:
        series_set = load_transformed_series_set(
            short_origin_set_name=orig_set_name, short_transform_name=transform_name
        )

    # ensure names in series set and results set are of the same type
    if type(series_set.series_set[0].name) != type(
        results_set.series_results[0].ts_name
    ):
        raise AssertionError

    # construct results df
    series_set_dict = series_set.get_series_dict()
    rows_dict: dict[str, Any] = dict()
    for ts_result in results_set.series_results:
        ts_dict = dict()
        ts = series_set_dict[ts_result.ts_name]
        for field_name, field_callable in fields_funs.items():
            ts_dict[field_name] = field_callable(ts, ts_result)
        rows_dict[ts.name] = ts_dict
    results_df = pd.DataFrame.from_dict(rows_dict, orient="index")

    return {
        "original_series_set": orig_set_name,
        "transform": transform_name,
        "method": results_set.method_name,
        "results_df": results_df,
    }
