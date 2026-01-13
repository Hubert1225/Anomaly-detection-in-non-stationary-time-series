"""This module provides implementation of the Multi-Window-Finder method
for domain agnostic window size estimation for unsupervised time
series analysis

Code obtained from:
https://sites.google.com/view/multi-window-finder/

References:
    Imani S, Abdoli A, Beyram A, Imani A, Keogh E. Multi-Window-Finder: Domain Agnostic Window Size
        for Time Series Data; 2021.

"""

import numpy as np
import pandas as pd


def windowLocator(window_sizes, movingAvgResiduals):
    t1 = np.linspace(0, 0.005, num=5)[::-1]
    t2 = np.linspace(-0.005, 0, num=5)
    thresholds = np.concatenate((t1, t2))
    for threshold in thresholds:
        for i in range(len(window_sizes) - 1):
            if movingAvgResiduals[i + 1] > (movingAvgResiduals[i] + threshold):
                return window_sizes[i]

    return "can not find the good window size"


def zscore(ts):
    ts = (ts - np.nanmean(ts)) / (np.nanstd(ts))
    return ts


def movmean(ts, w):
    """
    # faster solution of moving ave
    moving_avg = np.cumsum(ts, dtype=float)
    moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
    return moving_avg[w-1:] / w
    """

    moving_avg = np.cumsum(ts, dtype=float)
    moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
    return moving_avg[w - 1 :] / w


def original_finding_window_size(ts, start, end):
    """
    finidng appropriate window size using movemean
    """
    threshold = float("inf")
    all_averages = []
    window_sizes = []
    # step size was 50
    for w in range(start, end, 10):
        movingAvg = np.array(movmean(ts, w))

        all_averages.append(movingAvg)
        window_sizes.append(w)

    movingAvgResiduals = []

    for i, w in enumerate(window_sizes):
        moving_avg = all_averages[i][: len(all_averages[-1])]

        movingAvgResidual = np.log(abs(moving_avg - (moving_avg).mean()).sum())

        movingAvgResiduals.append(movingAvgResidual)

    b = (np.diff(np.sign(np.diff(movingAvgResiduals))) > 0).nonzero()[
        0
    ] + 1  # local min
    reswin = []
    try:
        for i in range(3):
            reswin.append(window_sizes[b[i]] / (i + 1))
        reswin = np.array(reswin)
        winTime = 0.8 * reswin[0] + 0.15 * reswin[1] + 0.05 * reswin[2]
        #         winTime = np.mean(reswin)
        conf = np.std(reswin) / np.sqrt(3)
    except:
        winTime = window_sizes[b[0]]
        conf = np.nan
    #     kn =  windowLocator(window_sizes, movingAvgResiduals) # window size (the reason I choose kn is because when I wanted to be the same name as kneelocator function)
    #     movingAvgResiduals = list(np.array(movingAvgResiduals[:-1]) - np.array(movingAvgResiduals[1:]))

    return winTime, conf, reswin


def combining_ts(paths):
    combined_time_series = np.array([])
    for file_path in paths:

        df = pd.read_csv(file_path, names=["values"])

        if df.shape == (1, 1):
            time_series = df["values"][0]
            time_series = time_series.split()
            for i in range(len(time_series)):
                time_series[i] = float(time_series[i])
            time_series = np.array(time_series)
        else:
            time_series = df["values"].values
        time_series = (time_series - np.min(time_series)) / (
            np.max(time_series) - np.min(time_series)
        )

        if len(combined_time_series) != 0:
            last_value = combined_time_series[-1]

            time_series = time_series - time_series[0] + last_value + 0.3

        combined_time_series = np.concatenate((combined_time_series, time_series))
    return combined_time_series


def finding_window_size_v1(ts, start, end):
    """
    finidng appropriate window size using movemean
    """
    threshold = float("inf")
    all_averages = []
    window_sizes = []
    # step size was 50
    for w in range(start, end, 10):
        movingAvg = np.array(movmean(ts, w))

        all_averages.append(movingAvg)
        window_sizes.append(w)

    movingAvgResiduals = []

    for i, w in enumerate(window_sizes):
        moving_avg = all_averages[i][: len(all_averages[-1])]

        movingAvgResiduals.append(moving_avg)

    return movingAvgResiduals, window_sizes
