"""This script provides helper functions to perform
anomaly detection with desired method on desired dataset
"""

from tqdm import tqdm
from typing import Any
import numpy as np
import pandas as pd

from base import (
    SubsequenceAnomalyDetector,
    DetectionResultForSeries,
    SeriesSet,
    SeriesSetResult,
)
from detectors.classical import STOMPDetector, NORMADetector, GraphTSDetector
from detectors.deep import TCNAEDetector
from detectors.baseline import RandomDetector
from results_utils import add_seasonality_column


def get_detector(
    method_name: str,
    method_params: dict[str, Any],
    anom_len: int,
    verbose: bool = True,
    **kwargs,
) -> SubsequenceAnomalyDetector:
    # if anom_len < 3, set it to 3
    anom_len_corrected = anom_len if anom_len >= 3 else 3

    if method_name == "random":
        detector = RandomDetector(anom_len=anom_len_corrected)
    elif method_name == "stomp":
        detector = STOMPDetector(anom_len=anom_len_corrected)
    elif method_name == "norma":
        detector = NORMADetector(
            anom_len=anom_len_corrected,
            cut_method=method_params["cut_method"],
            nm_multiplier=method_params["nm_multiplier"],
            percentage_sel=method_params["percentage_sel"],
            verbose=verbose,
        )
    elif method_name == "graphts":
        mwf_window_size = kwargs["mwf_window_size"]
        if (
            mwf_window_size is not None
            and mwf_window_size <= method_params["max_normal_pattern_len"]
        ):
            try:
                detector = GraphTSDetector(
                    l=anom_len_corrected,
                    l_np=mwf_window_size,
                    div=method_params["div"],
                    subtr=method_params["subtr"],
                )
            except GraphTSDetector.InvalidWg:
                detector = GraphTSDetector(
                    l=anom_len_corrected,
                    wg=method_params["wg_default"],
                )
        else:
            detector = GraphTSDetector(
                l=anom_len_corrected,
                wg=method_params["wg_default"],
            )
    elif method_name == "tcnae":
        detector = TCNAEDetector(
            anom_len=anom_len_corrected,
            kernel_size=method_params["kernel_size"],
            base_lr=method_params["lr"],
            ae_lr=method_params["lr"],
            loss_fun=method_params["loss_fun"],
            anom_windows_extraction_method=method_params[
                "anom_windows_extraction_method"
            ],
            errors_baseline_correct=method_params["errors_baseline_correct"],
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported method name: {method_name}")
    return detector


def detect_on_series_set(
    series_set: SeriesSet,
    series_set_desc: pd.DataFrame,
    method_name: str,
    method_params: dict[str, Any],
    quiet: bool = False,
    method_verbose: bool = True,
) -> SeriesSetResult:
    series_set_desc = add_seasonality_column(series_set_desc)
    series_results: list[DetectionResultForSeries] = []
    for ts in series_set.series_set if quiet else tqdm(series_set.series_set):

        if method_name == "graphts":
            kwargs = {
                "mwf_window_size": series_set_desc.loc[ts.name, "mwf_window_size"]
            }
        else:
            kwargs = {}

        # create detector for ts
        detector = get_detector(
            method_name=method_name,
            method_params=method_params,
            anom_len=int(series_set_desc.loc[ts.name, "anomaly length"]),
            verbose=method_verbose,
            **kwargs,
        )
        # detect anomalies
        detector.fit(ts.values)
        detected_anoms = detector.get_k_anoms(k=ts.anoms.shape[0])
        # create result object and add to list
        series_results.append(
            DetectionResultForSeries(
                ts_name=ts.name,
                method_name=method_name,
                detected_anoms=detected_anoms,
            )
        )
    return SeriesSetResult(
        set_name=series_set.name, method_name=method_name, series_results=series_results
    )
