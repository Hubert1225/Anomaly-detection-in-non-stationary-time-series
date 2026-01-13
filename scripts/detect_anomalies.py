import datetime
import os
import pandas as pd

from base import SeriesSetResult
from params import params, methods_params
from data_loading import load_series_set, load_transformed_series_set
from results_utils import load_series_descriptions
from detection_utils import detect_on_series_set

import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


set_names = params["detect_anomalies"]["sets"]
methods_names = params["detect_anomalies"]["methods"]
transform_name = params["detect_anomalies"]["transform"]

now = datetime.datetime.now()
results_dirname = (
    f"results_{transform_name}_{now.year}-{now.month}-{now.day}_{now.hour}_{now.minute}"
)
RESULTS_CATALOGUE = os.path.join("results", "detect_anomalies", results_dirname)
os.makedirs(RESULTS_CATALOGUE, exist_ok=False)


def save_series_set_results(set_results: SeriesSetResult) -> None:
    results_dir_path = os.path.join(
        RESULTS_CATALOGUE, f"{set_results.set_name}__{set_results.method_name}"
    )
    os.makedirs(results_dir_path, exist_ok=False)
    set_results.save(results_dir_path)


if __name__ == "__main__":

    print(f"Running detection for transform_name={transform_name}")

    for method_name in methods_names:
        print(method_name)
        for set_name in set_names:
            print(set_name)

            # load series set and its description
            if transform_name == "no_transformation":
                series_set = load_series_set(set_name)
            else:
                series_set = load_transformed_series_set(set_name, transform_name)
            series_set_desc = load_series_descriptions(set_name=set_name)
            series_set_desc.index = series_set_desc.index.astype("str")

            # get params for method, for set
            params_of_method = methods_params[transform_name][method_name][set_name]
            print(f"Parameters: {params_of_method}")

            # detect anomalies in each series and get overall result
            set_result = detect_on_series_set(
                series_set=series_set,
                method_name=method_name,
                method_params=params_of_method,
                series_set_desc=series_set_desc,
                method_verbose=False,
            )
            save_series_set_results(set_result)
