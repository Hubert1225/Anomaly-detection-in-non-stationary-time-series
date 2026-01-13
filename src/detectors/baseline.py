import numpy as np

from base import SubsequenceAnomalyDetector
from utils import get_k_max_nonoverlapping


class RandomDetector(SubsequenceAnomalyDetector):

    def __init__(self, anom_len: int):
        super().__init__(name="random")
        self.anom_len: int = anom_len
        self.inds = None

    def _fit(self, X: np.ndarray) -> None:
        self.inds = np.random.permutation(X.shape[0] - self.anom_len + 1)

    def _get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        anoms_inds = get_k_max_nonoverlapping(self.inds, window_len=self.anom_len, k=k)
        return [
            (int(anom_ind), int(anom_ind + self.anom_len)) for anom_ind in anoms_inds
        ]
