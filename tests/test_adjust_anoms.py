import numpy as np
import pandas as pd

from utils import adjust_invalid_anoms


def test_adjust_invalid_anoms():

    # TEST 1.
    input_anom_df = pd.DataFrame(
        {"anom_start": [0, 10, 20], "anom_end": [4, 14, 24]},
    )
    expected_anom_df = input_anom_df
    output_anom_df = adjust_invalid_anoms(anom_df=input_anom_df, ts_name="TEST1")
    if np.any(expected_anom_df != output_anom_df):
        raise AssertionError
    print(expected_anom_df)
    print(output_anom_df)

    # TEST 2.
    input_anom_df = pd.DataFrame(
        {"anom_start": [7, 15, -2], "anom_end": [11, 19, 2]},
    )
    expected_anom_df = pd.DataFrame(
        {"anom_start": [7, 15, 0], "anom_end": [11, 19, 4]},
    )
    output_anom_df = adjust_invalid_anoms(anom_df=input_anom_df, ts_name="TEST2")
    if np.any(expected_anom_df != output_anom_df):
        raise AssertionError
    print(expected_anom_df)
    print(output_anom_df)
