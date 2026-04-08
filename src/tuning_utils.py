"""This module provides tool useful for methods' hyperparameters
tuning
"""

from typing import Any
from tqdm import tqdm
import datetime as dt

import numpy as np
import pandas as pd
from numpy.random import default_rng

from base import SeriesSet, SeriesSetResult
from data_loading import load_series_set, load_transformed_series_set
from evaluation import get_recall_at_k
from detection_utils import detect_on_series_set


def get_random_series_set_subset(
    set_name: str,
    n_series: int,
    random_seed: int,
    transform_name: str,
) -> SeriesSet:

    if transform_name == "no_transformation":
        series_set = load_series_set(set_name)
    else:
        series_set = load_transformed_series_set(
            short_origin_set_name=set_name, short_transform_name=transform_name
        )

    if n_series > len(series_set.series_set):
        raise ValueError(
            f"Given value of `n_series`: {n_series} is bigger than number of all series in the series set: {len(series_set.series_set)}"
        )

    rng = default_rng(random_seed)
    chosen_indices = rng.choice(
        len(series_set.series_set), size=n_series, replace=False
    )

    return SeriesSet(
        series_set=[series_set.series_set[idx] for idx in chosen_indices],
        name=(series_set.name + f"_subset_{n_series}_{random_seed}"),
    )


def series_set_recall_at_k(
    series_set: SeriesSet,
    series_set_results: SeriesSetResult,
    min_part_detected: float,
):
    """Given detection results for a series set, calculates recall@k for each series
    and returns mean over all series
    """
    series_dict = series_set.get_series_dict()
    return np.mean(
        [
            get_recall_at_k(
                ts=series_dict[ts_result.ts_name],
                ts_result=ts_result,
                min_part_detected=min_part_detected,
            )
            for ts_result in series_set_results.series_results
        ]
    )


def method_grid_search_on_series_set(
    series_set: SeriesSet,
    series_set_desc: pd.DataFrame,
    method_name: str,
    min_part_detected: float,
    param_1_name: str,
    param_1_vals: list[Any],
    param_2_name: str,
    param_2_vals: list[Any],
    **kwargs,
) -> pd.DataFrame:

    results_df = pd.DataFrame(
        index=param_1_vals,
        columns=param_2_vals,
        data=np.full((len(param_1_vals), len(param_2_vals)), fill_value=-1.0),
    )

    for i, p1_val in enumerate(tqdm(param_1_vals, desc=param_1_name)):
        for j, p2_val in enumerate(tqdm(param_2_vals, desc=param_2_name)):

            result = detect_on_series_set(
                series_set=series_set,
                series_set_desc=series_set_desc,
                method_name=method_name,
                method_params={param_1_name: p1_val, param_2_name: p2_val, **kwargs},
                quiet=True,
                method_verbose=False,
            )

            results_df.iloc[i, j] = series_set_recall_at_k(
                series_set=series_set,
                series_set_results=result,
                min_part_detected=min_part_detected,
            )

    with open(f"tuning-{dt.datetime.now().strftime('%m-%d')}", "a") as f:
        f.write(f"{dt.datetime.now().strftime('%m-%d %H:%M:%S')}\n")

    return results_df


def combine_multiple_searches(
    results_dfs: list[pd.DataFrame],
    presentation: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:

    df_concat = pd.concat(results_dfs)
    df_groupby = df_concat.groupby(df_concat.index)
    df_means = df_groupby.mean()
    df_stds = df_groupby.std(ddof=0)

    if presentation:
        return df_means.round(3).astype("str") + " ± " + df_stds.round(3).astype("str")

    else:
        return df_means, df_stds
