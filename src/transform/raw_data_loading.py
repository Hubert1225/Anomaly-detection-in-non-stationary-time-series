"""This module provides functions that load
series set objects from downloaded raw data
"""

import os
import re
import numpy as np
import pandas as pd
import wfdb

from base import TimeSeriesWithAnoms, SeriesSet
from utils import point_to_subseq_annotations, RejectSeries


ucr_fname_regex = re.compile(r"^(\d+)_UCR_Anomaly_([\w\d]+)_(\d+)_(\d+)_(\d+)\.txt$")

files_tabsep = ["204", "205", "206", "207", "208", "225", "226", "242", "243"]

srw_regex = re.compile(
    r"^SinusRW_Length_(\d+)_AnomalyL_(\d+)_AnomalyN_(\d+)_NoisePerc_\d+$"
)


def load_ucr_series(file_path: str) -> TimeSeriesWithAnoms:
    """Loads a time series from file
    from UCR Time Series Anomaly Datasets 2021

    Args:
        file_path: path to the file with time series values
            to load

    Returns:
        TimeSeriesWithAnoms object
    """

    # parse filename
    file_name = os.path.basename(file_path)
    matches = ucr_fname_regex.findall(file_name)
    if len(matches) != 1:
        raise RuntimeError(
            f"UCR filename incorrectly parsed. Filename: {file_name}, matches: {matches}"
        )
    dataset_number, dataset_name, _, anom_start, anom_end = matches[0]
    anom_start = int(anom_start)
    anom_end = int(anom_end)

    # read data fram file
    if dataset_number in files_tabsep:
        with open(file_path) as f:
            values_lines = f.readlines()
        if len(values_lines) != 1:
            raise RuntimeError(
                f"Unknown file type encountered while reading {file_path}"
            )
        values = np.array([float(val) for val in values_lines[0].split()])
    else:
        values = pd.read_csv(file_path, header=None)[0].to_numpy()

    # return series object
    return TimeSeriesWithAnoms(
        values=values,
        anoms=[(anom_start, anom_end + 1)],
        name=f"{dataset_number}_{dataset_name}",
    )


def load_ucr_series_set(
    files_path: str, max_anom_len: int, max_len: int | None = None
) -> SeriesSet:
    """Loads all series from UCR Time Series Anomaly Datasets 2021
    and creates SeriesSet object

    Args:
        files_path: path to the directory with all UCR datasets files
        max_anom_len: maximal anomaly length, if a series' anomaly is longer,
            the series is rejected
        max_len: maximum length of a series, longer series are ejected;
            if None (default), no series are ejected

    Returns:
        SeriesSet object
    """

    ucr_files = os.listdir(files_path)

    # sort files names
    ucr_files.sort()

    # load series objects from files
    series_set = [
        load_ucr_series(os.path.join(files_path, fname)) for fname in ucr_files
    ]

    # reject too long series
    if max_len:
        series_set = [ts for ts in series_set if ts.values.shape[0] <= max_len]

    # reject if anomaly's too long
    series_set = [ts for ts in series_set if ts.anoms_lens().iloc[0] <= max_anom_len]

    return SeriesSet(series_set=series_set, name="UCR")


def get_mitbih_series(
    db_path: str, record_name: str, sampto=None
) -> TimeSeriesWithAnoms:
    """Loads a time series from MIT-BIH Arrhythmia Database
    or MIT-BIH Supraventicular Arrhythmia Database

    Args:
        db_path: path to the directory with MIT-BIH files
        record_name: record to be loaded
        sampto: the sample number at which to stop reading; reads the entire duration by default

    Returns:
        TimeSeriesWithAnoms object

    """

    try:
        record = wfdb.rdsamp(f"{db_path}/{record_name}", sampto=sampto)
        ann = wfdb.rdann(f"{db_path}/{record_name}", "atr", sampto=sampto)
    except ValueError:
        # sampto longer than signal length
        record = wfdb.rdsamp(f"{db_path}/{record_name}")
        ann = wfdb.rdann(f"{db_path}/{record_name}", "atr")

    values = record[0][:, 0]

    anoms = point_to_subseq_annotations(
        point_anns=ann.symbol,
        anns_inds=ann.sample,
        normal_symbol="N",
        series_len=values.shape[0],
    )

    return TimeSeriesWithAnoms(values, anoms, name=record_name)


def load_mitbih_series_set(
    db_path: str,
    name: str,
    max_contamination: float,
    max_anom_len: int,
    sampto=None,
) -> SeriesSet:
    """Loads SeriesSet from MIT-BIH files
    (MIT-BIH Arrhythmia Database or MIT-BIH Supraventricular Arrhythmia Database)

    Args:
        db_path: path to the directory with MIT-BIH files
        name: name that will be assigned to the series set
        max_contamination: float number from range (0.0, 1.0); series
            with anomalous part more than it will be dropped
        max_anom_len: maximum length of anomalous sequence
        sampto: the sample number at which to stop reading for each series;
            reads the entire duration by default
    """

    # load records list
    with open(os.path.join(db_path, "RECORDS")) as f:
        records = [fname.strip() for fname in f.readlines()]

    # load series objects
    series_set = []
    for record in records:
        try:
            ts = get_mitbih_series(db_path=db_path, record_name=record, sampto=sampto)
            series_set.append(ts)
        except RejectSeries:
            pass

    series_set = [
        ts
        for ts in series_set
        if (ts.anoms_lens().sum() / ts.values.shape[0]) <= max_contamination
    ]

    series_set = [ts for ts in series_set if ts.anoms_lens().iloc[0] <= max_anom_len]

    return SeriesSet(series_set=series_set, name=name)


def get_srw_series(dir_path: str, series_name: str) -> TimeSeriesWithAnoms:
    """Loads a time series from synthetic datasets corpus
    provided by NormA website

    Args:
        dir_path: path to the directory with
            files
        series_name: name of the series file without
            extension, e.g. `SinusRW_Length_120000_AnomalyL_200_AnomalyN_100_NoisePerc_0`

    Returns:
        TimeSeriesWithAnoms object

    """
    series_metadata_vals = srw_regex.findall(series_name)[0]
    series_length, anom_length, n_anoms = [int(val) for val in series_metadata_vals]

    series_path = os.path.join(dir_path, series_name + ".ts")
    ann_path = os.path.join(dir_path, series_name + "_Annotations.txt")

    with open(ann_path) as f:
        ann_inds = [int(line.strip()) for line in f.readlines()]
    anoms = [(ind, ind + anom_length) for ind in ann_inds]

    values = pd.read_csv(series_path, header=None).values.reshape(-1)

    return TimeSeriesWithAnoms(values=values, anoms=anoms, name=series_name)


def load_srw_series_set(dir_path: str, max_anom_len: int) -> SeriesSet:
    """Loads series set from synthetic datasets corpus
    provided by NormA website

    Args:
        dir_path: path to the directory with files
        max_anom_len: maximum length of anomalous subsequence

    Returns:
        SeriesSet object
    """

    # make list of all series to load
    series_names = set(
        [
            os.path.splitext(fname)[0]
            for fname in os.listdir(dir_path)
            if fname.endswith(".ts")
        ]
    )

    # load all series
    series_set = [
        get_srw_series(dir_path=dir_path, series_name=name) for name in series_names
    ]

    # remove series with too big contamination
    series_set = [ts for ts in series_set if ts.anoms_lens().iloc[0] <= max_anom_len]

    return SeriesSet(series_set=series_set, name="SRW")
