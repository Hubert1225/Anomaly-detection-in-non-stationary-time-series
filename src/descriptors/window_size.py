from typing import Any

from base import TimeSeriesDescriptor, TimeSeriesWithAnoms
from descriptors.multi_window_finder import original_finding_window_size


class MWFWindowSize(TimeSeriesDescriptor):

    def __init__(self, min_size: int, max_size: int):
        self.min_size: int = min_size
        self.max_size: int = max_size

    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:
        try:
            window_size = round(
                original_finding_window_size(
                    ts=ts.values, start=self.min_size, end=self.max_size
                )[0]
            )
        except IndexError:
            # MWF did not find good window size
            window_size = None
        return {"mwf_window_size": window_size}
