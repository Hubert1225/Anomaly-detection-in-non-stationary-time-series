"""This module provides tools to conveniently
visualize time series data and experiments results
"""

import numpy as np


def visualize_with_mark(
    values: np.ndarray, mark_inds: tuple[int, int], ax, color: str = "r"
) -> None:
    """Plots series of values with some region marked

    Args:
        values: all values to plot (1D array)
        mark_inds: (first_indx, last_indx + 1), where first_indx is the first
            and last_indx is the last index of the marked region
        ax: matplotlib Axes to plot on
        color: color of the marked region

    """

    x = np.arange(values.shape[0])
    x_marked = np.arange(mark_inds[0], mark_inds[1])
    values_marked = values[mark_inds[0] : mark_inds[1]]

    ax.plot(x, values)
    ax.plot(x_marked, values_marked, c=color)
