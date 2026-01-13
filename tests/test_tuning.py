import numpy as np
import pandas as pd

from tuning_utils import combine_multiple_searches


def are_values_close(val_1, val_2, tol) -> bool:
    return np.abs(val_1 - val_2) < tol


def test_combine_multiple_searches():

    # GIVEN
    index = [f"row_{i}" for i in range(4)]
    columns = [f"col_{i}" for i in range(2)]
    df1 = pd.DataFrame(
        index=index,
        columns=columns,
        data=np.array(
            [
                [0.0, 1.0],
                [2.0, -1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
    )
    df2 = pd.DataFrame(
        index=index,
        columns=columns,
        data=np.array(
            [
                [0.0, 2.0],
                [-2.0, -2.0],
                [0.0, 3.0],
                [-2.0, 1.0],
            ]
        ),
    )
    df3 = pd.DataFrame(
        index=index,
        columns=columns,
        data=np.array(
            [
                [0.0, 3.0],
                [-3.0, 4.0],
                [1.0, 2.0],
                [-2.0, 0.2],
            ]
        ),
    )
    tolerance = 1e-9

    # WHEN
    df_means, df_stds = combine_multiple_searches([df1, df2, df3], presentation=False)

    # THEN
    assert np.all(df_means.index == index)
    assert np.all(df_stds.index == index)
    assert np.all(df_means.columns == columns)
    assert np.all(df_stds.columns == columns)

    assert are_values_close(df_means.loc["row_0", "col_0"], 0.0, tol=tolerance)
    assert are_values_close(df_means.loc["row_0", "col_1"], 2.0, tol=tolerance)
    assert are_values_close(df_means.loc["row_1", "col_0"], -1.0, tol=tolerance)
    assert are_values_close(df_means.loc["row_1", "col_1"], 1 / 3, tol=tolerance)
    assert are_values_close(df_means.loc["row_2", "col_0"], 2 / 3, tol=tolerance)
    assert are_values_close(df_means.loc["row_2", "col_1"], 2.0, tol=tolerance)
    assert are_values_close(df_means.loc["row_3", "col_0"], -5 / 3, tol=tolerance)
    assert are_values_close(df_means.loc["row_3", "col_1"], 0.4, tol=tolerance)

    assert are_values_close(df_stds.loc["row_0", "col_0"], 0.0, tol=tolerance)
    assert are_values_close(
        df_stds.loc["row_0", "col_1"], np.sqrt(2 / 3), tol=tolerance
    )
    assert are_values_close(
        df_stds.loc["row_1", "col_0"], np.sqrt(14 / 3), tol=tolerance
    )
    assert are_values_close(
        df_stds.loc["row_1", "col_1"], np.sqrt(186 / 27), tol=tolerance
    )
    assert are_values_close(
        df_stds.loc["row_2", "col_0"], np.sqrt(6 / 27), tol=tolerance
    )
    assert are_values_close(
        df_stds.loc["row_2", "col_1"], np.sqrt(2 / 3), tol=tolerance
    )
    assert are_values_close(
        df_stds.loc["row_3", "col_0"], np.sqrt(6 / 27), tol=tolerance
    )
    assert are_values_close(
        df_stds.loc["row_3", "col_1"], np.sqrt(0.56 / 3), tol=tolerance
    )
