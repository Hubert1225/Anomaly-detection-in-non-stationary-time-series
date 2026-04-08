import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.distance import mahalanobis


def get_labels_series(length: int, ones_periods: list[tuple[int, int]]) -> np.ndarray:
    """Returns binary series with ones
    in desired periods
    """
    series = np.zeros(length, dtype=np.uint8)
    for period_start, period_end in ones_periods:
        series[period_start:period_end] = 1
    return series


def check_anom_df_consistency(anom_df: pd.DataFrame) -> bool:
    if np.any(anom_df.values < 0):
        return False
    return True


def adjust_invalid_anoms(anom_df: pd.DataFrame, ts_name: str) -> pd.DataFrame:
    invalid_inds = anom_df["anom_start"] < 0
    n_invalid = invalid_inds.sum()
    if n_invalid == 0:
        return anom_df
    if n_invalid > 1:
        raise RejectSeries
    warnings.warn(
        f"While processing series {ts_name} an anomaly became invalid."
        f"It will be adjusted"
    )
    corrected_anoms: list[tuple[int, int]] = []
    for index, (anom_start, anom_end) in anom_df.iterrows():
        anom_len = anom_end - anom_start
        if anom_start < 0:
            corrected_anoms.append((0, anom_len))
        else:
            corrected_anoms.append((anom_start, anom_end))
    series_len_to_check = max([anom[1] for anom in corrected_anoms])
    if are_anomalies_overlapping(anoms=corrected_anoms, series_len=series_len_to_check):
        raise RejectSeries
    return pd.DataFrame(corrected_anoms, columns=["anom_start", "anom_end"])


def sliding_window(ts: np.ndarray, window_len: int) -> np.ndarray:
    """Extracts subsequent sliding windows
    of given series values

    Args:
        ts (1D array): series values
        window_len (int): length of the sliding window

    Returns:
        2D array: array with subsequent sliding windows,
            of shape (n_windows, window_len),
            where n_windows is equal to ts_len - window_len + 1
    """
    n_windows = ts.shape[0] - window_len + 1

    sliding_inds = np.tile(np.arange(window_len), (n_windows, 1)) + np.arange(
        n_windows
    ).reshape((-1, 1))

    return ts[sliding_inds]


def nonoverlap_sliding_windows(ts: np.ndarray, window_len: int) -> np.ndarray:
    n_windows = ts.shape[0] // window_len
    sliding_inds = np.tile(np.arange(window_len), (n_windows, 1)) + np.arange(
        0, n_windows * window_len, window_len
    ).reshape((-1, 1))
    return ts[sliding_inds]


class SkipIteration(Exception):
    pass


def get_k_max_nonoverlapping(
    inds_sorted: list[int],
    window_len: int,
    k: int,
) -> list[int]:
    k_max_inds = []

    for window_ind in inds_sorted:

        try:

            # check if window does not overlap with any windows in list
            for ind_in_list in k_max_inds:
                if np.abs(window_ind - ind_in_list) < window_len:
                    raise SkipIteration

            # no overlaps - add ind to the list
            k_max_inds.append(window_ind)

            # if k items in list - break
            if len(k_max_inds) == k:
                break

        except SkipIteration:
            pass

    return k_max_inds


def mv_normal(mean: np.ndarray, cov: np.ndarray):

    n = cov.shape[0]
    regularization_coef = 1e-7
    regularization_applied = False

    while True:
        try:
            mv = sp.stats.multivariate_normal(mean=mean, cov=cov)
            break
        except np.linalg.LinAlgError:
            cov = cov + np.eye(n) * regularization_coef
            regularization_coef = regularization_coef * 2
            regularization_applied = True

    if regularization_applied:
        warnings.warn("Obtained a singular matrix, applied regularization.")

    return mv


def get_k_gaus_anomolus_windows(
    recon_errors: np.ndarray, window_len: int, k: int
) -> list[int]:
    recon_error_windows = sliding_window(recon_errors, window_len=window_len)
    recon_error_means = np.mean(recon_error_windows, axis=0)
    recon_error_cov = np.cov(recon_error_windows, rowvar=False)
    mv = mv_normal(mean=recon_error_means, cov=recon_error_cov)
    windows_logpdf = mv.logpdf(recon_error_windows)
    windows_inds_sorted = np.argsort(windows_logpdf)
    return get_k_max_nonoverlapping(
        inds_sorted=windows_inds_sorted.tolist(), window_len=window_len, k=k
    )


def get_k_gaus_anomolus_windows_md(
    recon_errors: np.ndarray, window_len: int, k: int
) -> list[int]:
    """Recon errors of shape:
    (n_channels, sequence_len)
    """
    seq_len = recon_errors.shape[-1]
    recon_error_windows = np.stack(
        [
            recon_errors[:, i : (i + window_len)].reshape((-1,))
            for i in range(seq_len - window_len)
        ]
    )
    recon_error_means = np.mean(recon_error_windows, axis=0)
    recon_error_cov = np.cov(recon_error_windows, rowvar=False)
    mv = mv_normal(mean=recon_error_means, cov=recon_error_cov)
    windows_logpdf = mv.logpdf(recon_error_windows)
    windows_inds_sorted = np.argsort(windows_logpdf)
    return get_k_max_nonoverlapping(
        inds_sorted=windows_inds_sorted.tolist(), window_len=window_len, k=k
    )


class RejectSeries(Exception):
    pass


def point_to_subseq_annotations(
    point_anns: list[str] | np.ndarray,
    anns_inds: list[int] | np.ndarray,
    normal_symbol: str,
    series_len: int,
) -> list[tuple[int, int]]:
    """
    Args:
        point_anns: list or array of strings where each string is annotation
            symbol of a point
        anns_inds: indexes for annotations symbol from `point_anns`;
            `anns_inds[i]` is the index of annotation `point_anns[i]`
        normal_symbol: symbol indicating normal point (no anomaly)
        series_len: length of the entire series
    """
    # check if there are any normal annotations in series
    # if not - reject series
    if all(ann_symbol != normal_symbol for ann_symbol in point_anns):
        raise RejectSeries

    # for each subsequence of anomalous symbols in `point_anns`
    # get tuple of indices of anomalous symbols in `point_anns`
    # e.g. if 'N' is normal symbol
    # and given point_anns: ['N', 'N', 'A', 'N', 'A', 'A', 'N']
    # the result should be: [(2,), (4, 5)]
    anoms_inds_in_point_anns: list[tuple[int, ...]] = []
    before_was_anom = False
    cur_anom_ann_inds = []
    for i, ann_symbol in enumerate(point_anns):
        if ann_symbol == normal_symbol:
            # symbol indicating normal point
            if before_was_anom:
                # end of anomaly
                anoms_inds_in_point_anns.append(tuple(cur_anom_ann_inds))
                cur_anom_ann_inds = []
            before_was_anom = False
        else:
            # symbol indicating anomaly
            cur_anom_ann_inds.append(i)
            before_was_anom = True
            if i == (len(point_anns) - 1):
                anoms_inds_in_point_anns.append(tuple(cur_anom_ann_inds))
                cur_anom_ann_inds = []

    # for each such subsequence, get indices of neighboring normal points
    # in series and calculate mean length of the section
    anoms_var_length: list[tuple[int, int]] = []
    for anom_symbols_inds in anoms_inds_in_point_anns:
        min_ind = min(anom_symbols_inds)
        max_ind = max(anom_symbols_inds)
        anoms_var_length.append(
            (
                anns_inds[(min_ind - 1)] if min_ind > 0 else 0,
                (
                    anns_inds[((max_ind + 1))]
                    if max_ind < (len(anns_inds) - 1)
                    else series_len
                ),
            ),
        )
    var_length_mean = int(
        np.mean([anom_end - anom_start for anom_start, anom_end in anoms_var_length])
    )
    length_mean_half = var_length_mean // 2

    # for each anomaly subsequence in annotation points, get median of annotations indexes in series
    # and add a window of fix length around this index to anomalies
    fixed_length_anoms: list[tuple[int, int]] = []
    for anom_symbols_inds in anoms_inds_in_point_anns:
        anom_inds_in_series = [anns_inds[ind] for ind in anom_symbols_inds]
        anom_centre = int(np.median(anom_inds_in_series))
        anom_start = anom_centre - length_mean_half
        anom_end = anom_centre + length_mean_half + 1
        if anom_start < 0:
            fixed_length_anoms.append((0, 2 * length_mean_half + 1))
        elif anom_end > series_len:
            fixed_length_anoms.append(
                (series_len - (2 * length_mean_half + 1), series_len)
            )
        else:
            fixed_length_anoms.append((anom_start, anom_end))

    # check if no anomalies are overlapping
    anom_len = fixed_length_anoms[0][1] - fixed_length_anoms[0][0]
    if any(
        (anom_end - anom_start) != anom_len
        for anom_start, anom_end in fixed_length_anoms
    ):
        raise AssertionError
    labels_series = get_labels_series(series_len, fixed_length_anoms)
    if np.sum(labels_series) != anom_len * len(fixed_length_anoms):
        raise RejectSeries

    return fixed_length_anoms


def are_anomalies_overlapping(anoms: list[tuple[int, int]], series_len: int) -> bool:
    """
    Returns True if there is any pair of anomalies
    that have overlap, False otherwise
    """
    labels_series = get_labels_series(length=series_len, ones_periods=anoms)
    anoms_len_sum = sum([anom_end - anom_start for anom_start, anom_end in anoms])
    return int(np.sum(labels_series)) != int(anoms_len_sum)


def anoms_df_to_tuples(anoms_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Convert dataframe of anomalies to list
    of tuples: (anom_start, anom_end)
    """
    return [(anom_start, anom_end) for _, (anom_start, anom_end) in anoms_df.iterrows()]


def get_dfs_mean(dfs: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
    """Returns new dataframe which elements
    are elementwise means of given dataframes
    """
    df_concat = pd.concat(dfs)
    return df_concat.groupby(df_concat.index).mean()


def get_dfs_std(dfs: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
    """Returns new dataframe which elements
    are elementwise standard deviations of given dataframes
    """
    df_concat = pd.concat(dfs)
    return df_concat.groupby(df_concat.index).std(ddof=0)


def get_dfs_sum(dfs: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
    """Returns new dataframe which elements
    are elementwise sums of given dataframes
    """
    df_concat = pd.concat(dfs)
    return df_concat.groupby(df_concat.index).sum()


def mahalanobis_anomaly_score(X: np.ndarray) -> np.ndarray:
    """Anomaly scores for samples based on their
    Mahalanobis distance

    Args:
        X: 2D array of samples

    Returns:
        1D array with anomaly scores for samples in X

    """
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False, bias=True)
    cov_inv = None
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("Singular covariance matrix - using regularization")
        n, _ = cov.shape
        i = 0
        while cov_inv is None:
            try:
                cov_inv = np.linalg.inv(cov + np.identity(n) * 0.01 * np.power(2, i))
            except np.linalg.LinAlgError:
                i += 1
    return np.array([mahalanobis(x, mu, cov_inv) for x in X])
