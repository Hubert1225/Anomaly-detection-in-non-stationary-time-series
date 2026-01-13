import numpy as np
import pandas as pd

from utils import get_dfs_mean, get_dfs_std, get_dfs_sum
from evaluation import is_detection_different


def test_get_dfs_mean_std_sum():

    # create exemplary dfs
    df1 = pd.DataFrame(
        {"col1": [1, 12, 5], "col2": [1.5, 3.0, 4.5]}, index=["a", "b", "c"]
    )
    df2 = pd.DataFrame(
        {"col1": [10, 20, 30], "col2": [-0.5, 0.5, 1.5]}, index=["c", "b", "a"]
    )
    df3 = pd.DataFrame(
        {"col1": [2, 20, 5, 1], "col2": [1.5, 4.5, 0.5, 0.0]},
        index=["a", "b", "c", "d"],
    )

    # get mean and standard deviation dfs
    df_mean = get_dfs_mean((df1, df2, df3))
    df_std = get_dfs_std((df1, df2, df3))
    df_sum = get_dfs_sum((df1, df2, df3))

    print(df1)
    print(df2)
    print(df3)
    print("========")
    print(df_mean)
    print(df_std)
    print(df_sum)

    # check df mean
    if np.abs(df_mean.loc["a", "col1"] - 11.0) > 1e-6:  # 33 / 3
        raise AssertionError
    if np.abs(df_mean.loc["b", "col2"] - (8.0 / 3.0)) > 1e-6:
        raise AssertionError
    if np.abs(df_mean.loc["c", "col2"] - 1.5) > 1e-6:  # 4.5 / 3
        raise AssertionError
    if np.abs(df_mean.loc["d", "col1"] - 1.0) > 1e-6:
        raise AssertionError

    # check df std
    # ['a', 'col1']: sqrt(((11 - 1)^2 + (11 - 30)^2 + (11 - 2)^2)/3) = sqrt((100 + 361 + 81)/3) =~ 13.4412301024373
    if np.abs(df_std.loc["a", "col1"] - 13.4412301024373) > 1e-6:
        raise AssertionError
    # ['b', 'col2']: sqrt(((3.0 - (8.0/3.0))^2 + (0.5 - (8.0/3.0))^2 + (4.5 - (8.0/3.0))^2)/3) =
    #                = sqrt((0.3333333333333335^2 + (-2.1666666666666665)^2 + 1.8333333333333335^2)/3) =
    #                = sqrt((0.11111111111111122 + 4.694444444444444 + 3.3611111111111116)/3) =~ 1.649915822768611
    if np.abs(df_std.loc["b", "col2"] - 1.649915822768611) > 1e-6:
        raise AssertionError

    # check df sum
    if df_sum.loc["a", "col1"] != 33:
        raise AssertionError
    if df_sum.loc["b", "col2"] != 8.0:
        raise AssertionError


def test_is_detection_different():

    # TEST 1.
    detected_original = 1
    not_detected_original = 9
    detected_transform = 8
    not_detected_transform = 2
    alternative = "greater"

    result = is_detection_different(
        detected_original=detected_original,
        detected_transform=detected_transform,
        not_detected_original=not_detected_original,
        not_detected_transform=not_detected_transform,
        alternative=alternative,
    )
    print(f"TEST 1: {result}")
    if not result:
        raise AssertionError

    # TEST 2.
    detected_original = 1
    not_detected_original = 9
    detected_transform = 8
    not_detected_transform = 2
    alternative = "less"

    result = is_detection_different(
        detected_original=detected_original,
        not_detected_original=not_detected_original,
        detected_transform=detected_transform,
        not_detected_transform=not_detected_transform,
        alternative=alternative,
    )
    print(f"TEST 2.: {result}")
    if result:
        raise AssertionError

    # TEST 3. (from scipy docs)

    detected_original = 12
    not_detected_original = 3
    detected_transform = 7
    not_detected_transform = 8
    alternative = "less"

    result = is_detection_different(
        detected_original=detected_original,
        not_detected_original=not_detected_original,
        detected_transform=detected_transform,
        not_detected_transform=not_detected_transform,
        alternative=alternative,
    )
    print(f"TEST 3.: {result}")
    if not result:
        raise AssertionError

    # TEST 4.
    detected_original = 50
    not_detected_original = 50
    detected_transform = 51
    not_detected_transform = 49
    alternative = "greater"

    result = is_detection_different(
        detected_original=detected_original,
        not_detected_original=not_detected_original,
        detected_transform=detected_transform,
        not_detected_transform=not_detected_transform,
        alternative=alternative,
    )
    print(f"TEST 4.: {result}")
    if result:
        raise AssertionError
