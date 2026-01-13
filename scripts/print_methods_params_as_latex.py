import pandas as pd
from params import methods_params
import sys


if __name__ == "__main__":

    method = sys.argv[1]
    transforms_names = set(methods_params.keys())
    hyperparams_dict_for_method = {
        (transform, series_set): methods_params[transform][method][series_set]
        for transform in methods_params.keys()
        for series_set in methods_params[transform][method].keys()
    }

    hyperparams_df_for_method = pd.DataFrame.from_dict(
        data=hyperparams_dict_for_method,
        orient="index",
    )

    if method == "graphts":
        hyperparams_df_for_method = hyperparams_df_for_method.loc[
            :, ["wg_default", "div", "subtr"]
        ]

    hyperparameters_latex_for_method = hyperparams_df_for_method.to_latex(
        label=f"table:hyperparameters_{method}",
    )

    print(hyperparameters_latex_for_method)

