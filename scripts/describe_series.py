"""This script obtains a set of features for each time series
that describe its properties (e.g. length, number of anomalies,
statistical, nonstationarity types etc.)
"""

import os

from data_loading import load_series_set
from params import params
from base import SeriesSetAnalyzer
from descriptors.basic import BasicProperties, BasicStatistics, AnomaliesStatistics
from descriptors.nonstationarity import NonstationarityTypes
from descriptors.seasonality import Seasonality
from descriptors.window_size import MWFWindowSize

RESULTS_DIR = os.path.join("results", "describe_series")

sets_to_describe = params["describe_series"]["sets"]
min_season = params["describe_series"]["min_season"]
max_season = params["describe_series"]["max_season"]
detrend_window_len = params["describe_series"]["detrend_window_len"]
sign_level = params["describe_series"]["sign_level"]
adf_max_lag = params["describe_series"]["adf_max_lag"]
mwf_min_window_size = params["describe_series"]["mwf_min_window_size"]
mwf_max_window_size = params["describe_series"]["mwf_max_window_size"]


if __name__ == "__main__":

    for set_name in sets_to_describe:
        print(f"Processing {set_name} ...")

        series_set = load_series_set(set_name)

        analyzer = SeriesSetAnalyzer(
            [
                BasicProperties(),
                BasicStatistics(),
                AnomaliesStatistics(),
                NonstationarityTypes(
                    sign_level=sign_level,
                    adf_max_lag=adf_max_lag,
                ),
                Seasonality(
                    detrend_window_len=detrend_window_len,
                    min_season=min_season,
                    max_season=max_season,
                ),
                MWFWindowSize(
                    min_size=mwf_min_window_size,
                    max_size=mwf_max_window_size,
                ),
            ]
        )
        desc_df = analyzer.describe_series_set(series_set)

        desc_df.to_csv(os.path.join(RESULTS_DIR, "desc_" + series_set.name + ".csv"))
