import pandas as pd

from utils import anoms_df_to_tuples


def test_anoms_df_to_tuples():

    # TEST 1.
    input_anoms_df = pd.DataFrame(
        {"anom_start": [0, 15, 23, 30], "anom_end": [5, 20, 28, 35]}
    )
    expected_anoms_tuples = [(0, 5), (15, 20), (23, 28), (30, 35)]
    output_anoms_tuples = anoms_df_to_tuples(input_anoms_df)
    print(input_anoms_df)
    print(expected_anoms_tuples)
    print(output_anoms_tuples)
    if expected_anoms_tuples != output_anoms_tuples:
        raise AssertionError

    # TEST 2.
    input_anoms_df = pd.DataFrame({"anom_start": [6, 20, 33], "anom_end": [15, 29, 42]})
    expected_anoms_tuples = [
        (6, 15),
        (20, 29),
        (33, 42),
    ]
    output_anoms_tuples = anoms_df_to_tuples(input_anoms_df)
    print(input_anoms_df)
    print(expected_anoms_tuples)
    print(output_anoms_tuples)
    if expected_anoms_tuples != output_anoms_tuples:
        raise AssertionError
