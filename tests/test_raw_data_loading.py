import numpy as np

from utils import point_to_subseq_annotations


def test_preprocess_mitbih_anoms():

    # TEST 1.

    annotations = ["A", "N", "N", "N", "N", "A", "N", "N", "N", "A", "Q", "A", "N"]
    annotations_indexes = np.array([10, 15, 18, 21, 25, 29, 34, 40, 44, 48, 52, 55, 60])
    normal_symbol = "N"
    series_len = 61

    # anomalies indexes in annotations: (0,), (5,), (9, 10, 11)
    # var length anomalies: (0, 15), (25, 34), (44, 60)
    # mean length of var length anomaly: int(mean(15 + 9 + 16)) = 13
    # length_mean_half = 13 // 2 = 6

    expected_anomalies = [
        (10 - 6, 10 + 6 + 1),
        (29 - 6, 29 + 6 + 1),
        (52 - 6, 52 + 6 + 1),
    ]
    got_anomalies = point_to_subseq_annotations(
        point_anns=annotations,
        anns_inds=annotations_indexes,
        normal_symbol=normal_symbol,
        series_len=series_len,
    )

    print(expected_anomalies)
    print(got_anomalies)

    if expected_anomalies != got_anomalies:
        raise AssertionError

    # TEST 2.
    annotations = ["A", "N", "N", "N", "A", "A", "N", "N", "N", "Q", "A", "N", "N", "A"]
    annotations_indexes = np.array(
        [5, 12, 15, 20, 30, 35, 41, 43, 49, 55, 57, 60, 65, 76]
    )
    normal_symbol = "N"
    series_len = 80

    # anomalies indexes in annotations: (0,), (4, 5), (9, 10), (13,)
    # var length anomalies: (0, 12), (20, 41), (49, 60), (65, 80)
    # mean length of var length anomaly: int(mean(12 + 21 + 11 + 15)) = 14
    # length_mean_half = 14 // 2 = 7

    expected_anomalies = [(0, 15), (32 - 7, 32 + 7 + 1), (56 - 7, 56 + 7 + 1), (65, 80)]
    got_anomalies = point_to_subseq_annotations(
        point_anns=annotations,
        anns_inds=annotations_indexes,
        normal_symbol=normal_symbol,
        series_len=series_len,
    )
    print(expected_anomalies)
    print(got_anomalies)
    if expected_anomalies != got_anomalies:
        raise AssertionError
