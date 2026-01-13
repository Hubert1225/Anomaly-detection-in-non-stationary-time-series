"""This script loads downloaded raw data into
series sets objects and saves them so that they can be
conveniently loaded later
"""

import os

from params import params
from base import SeriesSet
from raw_data_loading import (
    load_ucr_series_set,
    load_mitbih_series_set,
    load_srw_series_set,
)

RAW_DATA_CATALOGUE = os.path.join("data", "raw")

SERIES_SETS_CATALOGUE = os.path.join("data", "series_sets")

sets_to_load = params["load_series_sets"]["sets"]
ucr_max_len = params["load_series_sets"]["ucr_max_len"]
ucr_max_anom_len = params["load_series_sets"]["ucr_max_anom_len"]
mitbih_sampto = params["load_series_sets"]["mitbih_sampto"]
mitbih_max_contamination = params["load_series_sets"]["mitbih_max_contamination"]
mitbih_max_anom_len = params["load_series_sets"]["mitbih_max_anom_len"]
srw_max_anom_len = params["load_series_sets"]["srw_max_anom_len"]


def save_series_set(series_set: SeriesSet) -> None:
    set_path = os.path.join(SERIES_SETS_CATALOGUE, series_set.name)
    os.makedirs(set_path, exist_ok=False)
    series_set.save(set_path)


if __name__ == "__main__":

    for set_name in sets_to_load:

        print(f"Processing {set_name}")

        if set_name == "ucr":
            ucr = load_ucr_series_set(
                os.path.join(RAW_DATA_CATALOGUE, "ucr"),
                max_anom_len=ucr_max_anom_len,
                max_len=ucr_max_len,
            )
            save_series_set(ucr)

        elif set_name == "mitbih_arrhythmia":
            mitbih_arrhythmia = load_mitbih_series_set(
                os.path.join(RAW_DATA_CATALOGUE, "mitbih_arrythmia"),
                "MITBIH_Arrhythmia",
                max_contamination=mitbih_max_contamination,
                max_anom_len=mitbih_max_anom_len,
                sampto=mitbih_sampto,
            )
            save_series_set(mitbih_arrhythmia)

        elif set_name == "mitbih_supra":
            mitbih_supra = load_mitbih_series_set(
                os.path.join(RAW_DATA_CATALOGUE, "mitbih_supra"),
                "MITBIH_Supraventricular",
                max_contamination=mitbih_max_contamination,
                max_anom_len=mitbih_max_anom_len,
                sampto=mitbih_sampto,
            )
            save_series_set(mitbih_supra)

        elif set_name == "srw":
            srw = load_srw_series_set(
                os.path.join(RAW_DATA_CATALOGUE, "srw"),
                max_anom_len=srw_max_anom_len,
            )
            save_series_set(srw)

        else:
            raise ValueError(f"Series set {set_name} not supported")
