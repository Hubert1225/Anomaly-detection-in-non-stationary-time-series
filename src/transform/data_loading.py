"""Utils to load data saved in files
during pipeline
"""

import os

from base import SeriesSet

SERIES_SETS_CATALOGUE = os.path.join("data", "series_sets")

SET_SHORT_NAME_TO_NAME = {
    "ucr": "UCR",
    "mitbih_arrhythmia": "MITBIH_Arrhythmia",
    "mitbih_supra": "MITBIH_Supraventricular",
    "srw": "SRW",
}

TRANSFORM_TO_SUFFIX = {
    "detrend": "_detrending",
    "simple_diff": "_differencing_1",
    "season_diff": "_differencing_dynamic",
    "season_diff_robust": "_robustdifferencing_dynamic_3",
    "rolling_mean": "_rollingmean_10",
    "box_cox": "_boxcox",
    "add_trend": "_addtrend",
}


def load_series_set(short_name: str) -> SeriesSet:
    """Loads a series set given its
    short name. Accepted series sets' short names:
    'ucr', 'mitbih_arrhythmia', 'mitbih_supra', 'srw'
    """
    set_name = SET_SHORT_NAME_TO_NAME[short_name]
    set_path = os.path.join(SERIES_SETS_CATALOGUE, set_name)
    return SeriesSet.load(set_path)


def load_transformed_series_set(
    short_origin_set_name: str, short_transform_name: str
) -> SeriesSet:
    """Loads a transformed version of a series set.

    Args:
        short_origin_set_name: short name of the series set which
            transformed version is to be loaded; possible:
            'ucr', 'mitbih_arrhythmia', 'mitbih_supra', 'srw'
        short_transform_name: short name of the transformation after
            which the set to load, possible:
            'detrend', 'simple_diff', 'season_diff', 'season_diff_robust',
            'rolling_mean', 'box_cox', 'add_trend'

    """
    original_set_name = SET_SHORT_NAME_TO_NAME[short_origin_set_name]
    transform_suffix = TRANSFORM_TO_SUFFIX[short_transform_name]
    full_set_name = original_set_name + transform_suffix
    set_path = os.path.join(SERIES_SETS_CATALOGUE, full_set_name)
    return SeriesSet.load(set_path)
