from typing import Literal

import numpy as np
import pandas as pd
from stumpy import stump

from base import SubsequenceAnomalyDetector

try:
    from .GraphTS import GraphTS
except ImportError:
    print(
        f"No GraphTS code! In order to run GraphTS, request the method's authors "
        f"for code: https://sites.google.com/view/graphts"
    )

try:
    from .norma import NORMA
except ImportError:
    print(
        f"No NormA code! In order to run NormA, request the method's authors "
        f"for code"
    )

from utils import get_k_max_nonoverlapping, are_anomalies_overlapping


class GraphTSDetector(SubsequenceAnomalyDetector):

    class InvalidWg(Exception):
        """Exception raised when `wg` calculated with given arguments
        is invalid (e.g. negative)
        """

        pass

    def __init__(
        self,
        l: int,
        wg: int | None = None,
        l_np: int | None = None,
        div: float | None = None,
        subtr: int | None = None,
        nc: int = 10,
    ):
        super().__init__(name="GraphTS")

        self.check_args(wg=wg, l_np=l_np, div=div, subtr=subtr)

        self.wg: int = wg or self._get_wg(l_np=l_np, div=div, subtr=subtr)

        if self.wg <= 2:
            raise self.InvalidWg

        self.nc: int = nc
        self.l: int = l
        self.values: pd.DataFrame | None = None

    @staticmethod
    def _get_wg(l_np: int, div: float, subtr: int) -> int:
        return round(l_np / div - subtr)

    @staticmethod
    def check_args(wg, l_np, div, subtr) -> None:
        wg_set_manually = (
            (wg is not None) and (l_np is None) and (div is None) and (subtr is None)
        )
        wg_to_calculate = (
            (wg is None)
            and (l_np is not None)
            and (div is not None)
            and (subtr is not None)
        )
        assert_mesage = (
            "you should either set `wg` and not none of the parameters: `l_np`, `div`, `subtr`, "
            f"or not set `wg` and set each of the parameters: `l_np`, `div`, `subtr`"
        )
        assert wg_set_manually or wg_to_calculate, assert_mesage

    def _fit(self, X: np.ndarray) -> None:
        self.values = pd.DataFrame(X)

    def _get_k_anoms_from_model(self, k: int) -> list[tuple[int, int]]:
        Score, l_index = GraphTS(
            self.values,
            wg=self.wg,
            l=self.l,
            k=k,
            nc=self.nc,
            TwoDSTS_image="No",
            GraphC="No",
        )
        return [(l_index[i], l_index[i] + self.l) for i in range(k)]

    def _get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        anoms = self._get_k_anoms_from_model(k)
        if len(anoms) != k:
            raise AssertionError
        if are_anomalies_overlapping(anoms=anoms, series_len=self.l):
            cur_k = k
            anoms_starts_nonoverlap = []
            while len(anoms_starts_nonoverlap) != k:
                cur_k += 1
                anoms = self._get_k_anoms_from_model(cur_k)
                anoms_starts = [anom[0] for anom in anoms]
                anoms_starts_nonoverlap = get_k_max_nonoverlapping(
                    anoms_starts, window_len=self.l, k=k
                )
            anoms_nonoverlap = [
                (anom_start, anom_start + self.l)
                for anom_start in anoms_starts_nonoverlap
            ]
        else:
            anoms_nonoverlap = anoms
        return anoms_nonoverlap


class STOMPDetector(SubsequenceAnomalyDetector):

    def __init__(self, anom_len: int):
        super().__init__(name="STOMP")
        self.anom_len: int = anom_len
        self.mp: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self.mp = stump(
            X,
            m=(self.anom_len if self.anom_len >= 3 else 3),
            ignore_trivial=True,
            normalize=True,
        )[:, 0]

    def _get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        anoms_inds = get_k_max_nonoverlapping(
            inds_sorted=np.flip(np.argsort(self.mp)).tolist(),
            window_len=self.anom_len,
            k=k,
        )
        return [(anom_ind, anom_ind + self.anom_len) for anom_ind in anoms_inds]


class NORMADetector(SubsequenceAnomalyDetector):

    def __init__(
        self,
        anom_len: int,
        nm_size: int | None = None,
        cut_method: Literal["max", "minmax", "auto"] = "max",
        nm_multiplier: float = 3.0,
        percentage_sel: float = 0.4,
        verbose: bool = True,
    ):
        super().__init__(name="NORMA")
        self.anom_len: int = anom_len
        pattern_length = anom_len if anom_len >= 3 else 3
        nm_size = nm_size if nm_size else int(nm_multiplier * pattern_length)
        self.model: NORMA = NORMA(
            pattern_length=pattern_length,
            nm_size=nm_size,
            cut_method=cut_method,
            percentage_sel=percentage_sel,
            verbose=verbose,
        )
        self.decision_scores: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self.decision_scores = self.model.decision_scores_

    def _get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        anoms_inds = get_k_max_nonoverlapping(
            inds_sorted=np.flip(np.argsort(self.decision_scores)).tolist(),
            window_len=self.anom_len,
            k=k,
        )
        return [(anom_ind, anom_ind + self.anom_len) for anom_ind in anoms_inds]
