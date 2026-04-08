"""This module provides tools to calculate
evaluation metrics on anomaly detection results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy.stats import barnard_exact

from base import TimeSeriesWithAnoms, DetectionResultForSeries
from utils import anoms_df_to_tuples, get_labels_series
from params import params
from utils import get_dfs_mean, get_dfs_sum
from results_utils import (
    get_results_df,
    load_series_descriptions,
    add_seasonality_column,
)

MIN_PART_DETECTED = params["evaluation"]["min_part_detected"]
DETECTION_RESULTS_BASE_PATH = os.path.join("results", "detect_anomalies")
EVALUATION_RESULTS_BASE_PATH = os.path.join("results", "evaluation")


def get_number_of_detected(
    ts: TimeSeriesWithAnoms,
    ts_result: DetectionResultForSeries,
    min_part_detected: float,
) -> int:
    """Calculates number of ground truth anomalies
    in given series that have been detected
    """
    ground_truth_anoms = anoms_df_to_tuples(ts.anoms)
    detected_anoms = anoms_df_to_tuples(ts_result.detected_anoms)
    series_length = ts.values.shape[0]
    if_detected: list[bool] = []
    detected_labels = get_labels_series(
        length=series_length, ones_periods=detected_anoms
    )
    for anom in ground_truth_anoms:
        anom_labels = get_labels_series(length=series_length, ones_periods=[anom])
        anom_length = anom[1] - anom[0]
        if_detected.append(
            (np.sum(np.logical_and(anom_labels, detected_labels)) / anom_length)
            > min_part_detected
        )
    return sum(if_detected)


def get_recall_at_k(
    ts: TimeSeriesWithAnoms,
    ts_result: DetectionResultForSeries,
    min_part_detected: float,
) -> float:
    n_detected = get_number_of_detected(
        ts,
        ts_result,
        min_part_detected,
    )
    n_ground_truth = ts.anoms.shape[0]
    return n_detected / n_ground_truth


def get_pointwise_f1score_at_k(
    ts: TimeSeriesWithAnoms,
    ts_result: DetectionResultForSeries,
) -> float:
    ground_truth_anoms = anoms_df_to_tuples(ts.anoms)
    detected_anoms = anoms_df_to_tuples(ts_result.detected_anoms)
    series_length = ts.values.shape[0]
    detected_labels = get_labels_series(
        length=series_length, ones_periods=detected_anoms
    )
    ground_truth_labels = get_labels_series(
        length=series_length, ones_periods=ground_truth_anoms
    )
    return f1_score(
        y_true=ground_truth_labels, y_pred=detected_labels, average="binary"
    )


def get_all_results_dict() -> dict[tuple[str, str, str], list[pd.DataFrame]]:
    """Loads all results from directories in each catalogue in
    DETECTION_RESULTS_BASE_PATH and runs `get_results_df` function on each results
    set directory (see results_utils module). Then, it creates dict:
    (original_set_name, transform_name, method_name) -> dfs_list
    where dfs_list is a list of all dataframes that comes from `get_results_df`
    with such an original_set_name, transform_name and method_name

    Dataframes contain following per-series metrics: number of detected anomalies,
    number of ground truth anomalies, recall at k and pointwise F1 score at k

    """

    def call_n_detected(ts, ts_result) -> int:
        return get_number_of_detected(
            ts=ts, ts_result=ts_result, min_part_detected=MIN_PART_DETECTED
        )

    def call_n_anoms(ts, ts_result) -> int:
        return ts.anoms.shape[0]

    def call_recall_at_k(ts, ts_result) -> float:
        return get_recall_at_k(
            ts=ts, ts_result=ts_result, min_part_detected=MIN_PART_DETECTED
        )

    def call_pw_f1score_at_k(ts, ts_result) -> float:
        return get_pointwise_f1score_at_k(ts=ts, ts_result=ts_result)

    fields_funs = {
        "n detected": call_n_detected,
        "n gf anomalies": call_n_anoms,
        "recall at k": call_recall_at_k,
        "pw f1 score at k": call_pw_f1score_at_k,
    }

    # dict: (orig_set_name, transform_name | 'original', method_name) -> list[result_df]
    all_results_dfs_dict: dict[tuple[str, str, str], list[pd.DataFrame]] = dict()

    for results_catalogue in tqdm(
        os.listdir(DETECTION_RESULTS_BASE_PATH), desc="catalogues"
    ):
        if results_catalogue == ".gitkeep":
            continue
        catalogue_path = os.path.join(DETECTION_RESULTS_BASE_PATH, results_catalogue)
        for results_set_dir in os.listdir(catalogue_path):
            result_set_path = os.path.join(catalogue_path, results_set_dir)
            result_dict = get_results_df(
                result_set_dir=result_set_path, fields_funs=fields_funs
            )
            orig_set_name = result_dict["original_series_set"]
            transform_name = (
                result_dict["transform"]
                if result_dict["transform"] is not None
                else "original"
            )
            method_name = result_dict["method"]
            if (
                orig_set_name,
                transform_name,
                method_name,
            ) in all_results_dfs_dict.keys():
                all_results_dfs_dict[
                    (orig_set_name, transform_name, method_name)
                ].append(result_dict["results_df"])
            else:
                all_results_dfs_dict[(orig_set_name, transform_name, method_name)] = [
                    result_dict["results_df"]
                ]
    return all_results_dfs_dict


def get_differences_df(
    results_df_original: pd.DataFrame,
    results_df_transformed: pd.DataFrame,
    transform_name: str,
) -> pd.DataFrame:
    """Returns dataframe of per-series metrics differences

    Args:
        results_df_original: dataframe with metrics for series from original (not transformed)
            series set
        results_df_transformed: dataframe with metrics for series from transformed series set
        transform_name: name of the transformation applied on transformed series set

    Returns:
        DataFrame: values are differences: value_for_transformed - value_for_original
    """
    rows_dict = dict()
    for ts_name in results_df_transformed.index:
        ts_diffs_dict = dict()
        for col_name in ["recall at k", "pw f1 score at k"]:
            ts_diffs_dict[f"{col_name} {transform_name} diff"] = (
                results_df_transformed.loc[ts_name, col_name]
                - results_df_original.loc[ts_name, col_name]
            )
        rows_dict[ts_name] = ts_diffs_dict
    return pd.DataFrame.from_dict(rows_dict, orient="index")


def get_results_table(
    all_results_dict: dict[tuple[str, str, str], list[pd.DataFrame]],
    transform_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns matrices of shape n_methods x n_series_sets
    with recall at k and pointwise F1 score at k for a method
    and a series set, with application of a transformation

    Args:
        all_results_dict: dict obtained with `get_all_results_dict` function
        transform_name: name of the transformatiom

    Returns:
        tuple: (recall_dataframe, f1_dataframe),
            where recall dataframe contains recall at k values
            (index are methods names and columns are series sets names),
            f1_dataframe analogously for pointwise F1 score at k

    """
    methods = ["random", "stomp", "graphts", "norma", "tcnae"]
    if transform_name == "add_trend":
        series_sets = ["ucr", "mitbih_arrhythmia", "mitbih_supra"]
    else:
        series_sets = ["ucr", "mitbih_arrhythmia", "mitbih_supra", "srw"]
    recalls_at_k = pd.DataFrame(
        data=np.full(shape=(len(methods), len(series_sets)), fill_value=-1.0),
        index=methods,
        columns=series_sets,
    )
    pw_f1scores_at_k = pd.DataFrame(
        data=np.full(shape=(len(methods), len(series_sets)), fill_value=-1.0),
        index=methods,
        columns=series_sets,
    )
    for method_name in methods:
        for set_name in series_sets:
            mean_results_df = get_dfs_mean(
                tuple(all_results_dict[(set_name, transform_name, method_name)])
            )
            recalls_at_k.loc[method_name, set_name] = mean_results_df[
                "recall at k"
            ].mean()
            pw_f1scores_at_k.loc[method_name, set_name] = mean_results_df[
                "pw f1 score at k"
            ].mean()
    return recalls_at_k, pw_f1scores_at_k


def get_differences_all_table(
    all_results_dict: dict[tuple[str, str, str], list[pd.DataFrame]],
    set_name: str,
    method_name: str,
) -> pd.DataFrame:
    """Given a series set and a method, returns dataframe with
    per-series metrics differences for all transformations

    Args:
        all_results_dict: dict obtained with `get_all_results_dict` function
        set_name: name of the series set to examine detection on series from
        method_name: name of the detection method

    Returns:
        dataframe: rows correspond to time series and columns to
            combinations metric - transformation

    """
    if set_name == "srw":
        transforms = [
            "rolling_mean",
            "detrend",
            "simple_diff",
            "season_diff",
            "box_cox",
            "season_diff_robust",
        ]
    else:
        transforms = [
            "rolling_mean",
            "detrend",
            "simple_diff",
            "season_diff",
            "box_cox",
            "season_diff_robust",
            "add_trend",
        ]
    results_df_original = get_dfs_mean(
        tuple(all_results_dict[(set_name, "original", method_name)])
    )
    diffs_dfs = []
    for transform_name in transforms:
        results_df_transform = get_dfs_mean(
            tuple(all_results_dict[(set_name, transform_name, method_name)])
        )
        diffs_dfs.append(
            get_differences_df(
                results_df_original=results_df_original,
                results_df_transformed=results_df_transform,
                transform_name=transform_name,
            )
        )
    return diffs_dfs[0].join(diffs_dfs[1:])


def is_detection_different(
    detected_original: int,
    not_detected_original: int,
    detected_transform: int,
    not_detected_transform: int,
    alternative: str,
    sign_level: float = 0.05,
) -> bool:
    # contingency table:
    #                    transformed       original
    #       n detected        .                .
    #   n not detected        .                .
    contingency_table = np.array(
        [
            [detected_transform, detected_original],
            [not_detected_transform, not_detected_original],
        ]
    )
    p_val = barnard_exact(
        contingency_table, alternative=alternative, pooled=False
    ).pvalue
    return p_val < sign_level


def get_statistical_difference_table(
    all_results_dict: dict[tuple[str, str, str], list[pd.DataFrame]],
    transform_name: str,
    method_name: str,
    alternative: str,
    sign_level: float = 0.05,
) -> pd.DataFrame:
    sets_names = [
        ("ucr", "UCR"),
        ("mitbih_arrhythmia", "MITBIH Arrhythmia"),
        ("mitbih_supra", "MITBIH Supraventricular"),
        ("srw", "SRW"),
    ]

    # get results for not transformed
    dfs_to_concat = []
    for short_set_name, long_set_name in sets_names:
        results_df = get_dfs_sum(
            tuple(all_results_dict[(short_set_name, "original", method_name)])
        )
        dfs_to_concat.append(results_df)
    all_sets_original_df = pd.concat(tuple(dfs_to_concat))
    all_sets_original_df.columns = [
        col_name + " original" for col_name in all_sets_original_df.columns
    ]

    # get results for transformed
    dfs_to_concat = []
    for short_set_name, long_set_name in sets_names:
        results_df = get_dfs_sum(
            tuple(all_results_dict[(short_set_name, transform_name, method_name)])
        )
        dfs_to_concat.append(results_df)
    all_sets_transform_df = pd.concat(tuple(dfs_to_concat))
    all_sets_transform_df.columns = [
        col_name + f" {transform_name}" for col_name in all_sets_transform_df.columns
    ]

    all_sets_df = all_sets_original_df.join(all_sets_transform_df)

    # check statistical significance for each time series
    all_sets_df["change"] = all_sets_df.apply(
        lambda row: is_detection_different(
            detected_original=row.loc["n detected original"],
            not_detected_original=(
                row.loc["n gf anomalies original"] - row.loc["n detected original"]
            ),
            detected_transform=row.loc[f"n detected {transform_name}"],
            not_detected_transform=(
                row.loc[f"n gf anomalies {transform_name}"]
                - row.loc[f"n detected {transform_name}"]
            ),
            alternative=alternative,
            sign_level=sign_level,
        ),
        axis=1,
    )

    return all_sets_df[
        [
            "n detected original",
            "n gf anomalies original",
            f"n detected {transform_name}",
            f"n gf anomalies {transform_name}",
            "change",
        ]
    ]


def detection_nonstats_comparison(
    all_results_dict: dict[tuple[str, str, str], list[pd.DataFrame]],
    method_name: str,
    metric: str,
    identifier: str,
) -> None:
    """
    metric: {"recall at k", "pw f1 score at k"}
    """
    transform_name = "original"
    sets_names = [
        ("ucr", "UCR"),
        ("mitbih_arrhythmia", "MITBIH Arrhythmia"),
        ("mitbih_supra", "MITBIH Supraventricular"),
        ("srw", "SRW"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6, 4), constrained_layout=True)
    joined_dfs = []
    for short_set_name, long_set_name in sets_names:
        results_df = get_dfs_mean(
            tuple(all_results_dict[(short_set_name, transform_name, method_name)])
        )
        set_desc = load_series_descriptions(short_set_name)
        set_desc = add_seasonality_column(set_desc)
        joined_dfs.append(results_df.join(set_desc))
    all_sets_joined_df = pd.concat(tuple(joined_dfs))
    sns.barplot(all_sets_joined_df.groupby("Stationary").mean()[metric], ax=axes[0])
    sns.barplot(
        all_sets_joined_df.groupby("Heteroscedasticity").mean()[metric],
        ax=axes[1],
    )
    sns.barplot(
        all_sets_joined_df.groupby("ClearSeasonality").mean()[metric], ax=axes[2]
    )
    for ax in axes:
        ax.set_ylim((0.0, 0.7))
    fig.suptitle(method_name)
    if metric == "recall at k":
        metric_name = "recall"
    elif metric == "pw f1 score at k":
        metric_name = "f1"
    else:
        raise AssertionError
    plt.savefig(
        os.path.join(
            EVALUATION_RESULTS_BASE_PATH,
            f"nonstat_comparison_{metric_name}_{method_name}_{identifier}.png",
        )
    )
