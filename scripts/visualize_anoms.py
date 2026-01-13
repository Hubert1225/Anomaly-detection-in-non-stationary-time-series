import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from base import TimeSeriesWithAnoms
from data_loading import load_series_set
from params import params

RESULTS_DIR = os.path.join("results", "visualize_anoms")

sets_to_visualize = params["visualize_anoms"]["sets"]
margin_coef = params["visualize_anoms"]["margin_coef"]
max_k_anoms = params["visualize_anoms"]["max_k_anoms"]
random_seed = params["general"]["random_seed"]


def get_max_margins(ts: TimeSeriesWithAnoms) -> tuple[pd.Series, pd.Series]:
    """Returns tuple of pandas Series: (max_left_margins, max_right_margins).

    max_left_margin at position i contains maximum left margin that can be used
    for i-th anomaly visualization. max_right_margin analogously, for the
    right margin
    """
    series_len = ts.values.shape[0]
    max_left_margins = ts.anoms["anom_start"]
    max_right_margins = series_len - ts.anoms["anom_end"]
    return max_left_margins, max_right_margins


def get_visualization_filename(
    ts: TimeSeriesWithAnoms, i: int, extension: str = "png"
) -> str:
    return f"tsanoms_{ts.name}_{i}.{extension}"


def visualize_ts_anoms(ts: TimeSeriesWithAnoms, output_dir: str) -> None:
    """Visualizes anomalies from ts (maximum max_k_anoms;
    if ts has more anomalies, max_k_anoms anomalies are chosen
    by random, with the set random seed). Visualizations are saved in output_dir
    """

    # if there is too many anoms, choose randomly
    n_ts_anoms = ts.anoms.shape[0]
    ts_anoms_inds = np.arange(n_ts_anoms)
    if n_ts_anoms > max_k_anoms:
        rng = np.random.default_rng(random_seed)
        chosen_inds = rng.choice(ts_anoms_inds, size=max_k_anoms, replace=False)
    else:
        chosen_inds = ts_anoms_inds

    # get anoms lens and max left, right margins
    anoms_lens = ts.anoms_lens()
    max_left_margins, max_right_margins = get_max_margins(ts)

    # visualize all (chosen) anomalies
    for anom_ind in chosen_inds:

        anom_len = anoms_lens.iloc[anom_ind]

        # get left and right margin
        left_margin = min(max_left_margins.iloc[anom_ind], int(anom_len * margin_coef))
        right_margin = min(
            max_right_margins.iloc[anom_ind], int(anom_len * margin_coef)
        )

        # create visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ts.visualize_anom(
            i=anom_ind,
            left_margin=left_margin,
            right_margin=right_margin,
            ax=ax,
            xticks_gran=10
            ** (round(np.log10(anom_len + left_margin + right_margin) - 1)),
        )
        fig.tight_layout()

        # save visualization to file
        visualization_path = os.path.join(
            output_dir, get_visualization_filename(ts, anom_ind)
        )
        plt.savefig(visualization_path)

        # close figure
        plt.close()


if __name__ == "__main__":

    for set_name in sets_to_visualize:

        # load series set
        series_set = load_series_set(set_name)

        # create subdirectory for the series set (error if already exists)
        output_dir = os.path.join(RESULTS_DIR, series_set.name)
        os.makedirs(output_dir, exist_ok=False)

        # visualize anomalies for each series in the series set and save to files
        for ts in tqdm(series_set.series_set, desc=set_name):
            visualize_ts_anoms(ts, output_dir)
