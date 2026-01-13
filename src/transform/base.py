"""Basic classes for the experiment pipeline
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import json
from typing import Any
from tqdm import tqdm

from visualization import visualize_with_mark
from utils import are_anomalies_overlapping, RejectSeries


class TimeSeriesWithAnoms:
    """Represents one univariate time series with annotated
    subsequence anomalies

    Args:
        values (1D array): Numpy array with values of the series
        anoms (list[tuple[int, int]] | pd.DataFrame): list of pairs (anom_start, anom_end),
            where anom_start is index of the first observation of an anomalous subsequence,
            and anom_end is (last_index_of_anom + 1)
            or a DataFrame, where each pair is one row and column are: 'anom_start', 'anom_end'

        name (str): the name of the time series, used to identify the series in a
            series set
    """

    def __init__(
        self, values: np.ndarray, anoms: list[tuple[int, int]] | pd.DataFrame, name: str
    ):

        self.__check_values(values)
        self.__check_anoms(anoms)

        self.values: np.ndarray = values
        self.anoms: pd.DataFrame = (
            anoms
            if isinstance(anoms, pd.DataFrame)
            else pd.DataFrame(anoms, columns=["anom_start", "anom_end"])
        )
        self.name: str = name

    def save(self, dir_path: str) -> None:
        """Saves the object into files so that it
        can be loaded later with `TimeSeriesWithAnoms.load()`

        Args:
            dir_path: path to the directory where to save the series
                files

        """
        values_filename, annotations_filename = self.__get_filenames_for_name(self.name)

        # save values
        np.save(file=os.path.join(dir_path, values_filename), arr=self.values)

        # save annotations (self.anoms)
        self.anoms.to_feather(os.path.join(dir_path, annotations_filename))

    def visualize_anom(
        self,
        i: int,
        left_margin: int,
        right_margin: int,
        ax,
        xticks_gran: int,
    ) -> None:
        """Plots a window of the series with anomalous
        subsequence marked

        Args:
            i: which anomaly to draw, 0 <= i < number_of_anoms
            left_margin: number of observations left to the anomaly to draw
            right_margin: number of observations right to the anomaly to draw
            ax: matplotlib Axes to plot on
            xticks_gran: distance bewteen xticklabels (granularity)

        """
        anom_start, anom_end = self.anoms.iloc[i].values
        ind_min = anom_start - left_margin
        if ind_min < 0:
            raise ValueError(
                "Too big left margin - insufficient data left to the anomaly"
            )
        ind_max = anom_end + right_margin
        if ind_max > self.values.shape[0]:
            raise ValueError(
                "Too big right margin - insufficient data right to the anomaly"
            )
        values_to_plot = self.values[ind_min:ind_max]
        anom_plot_start = anom_start - ind_min
        anom_plot_end = anom_end - ind_min
        visualize_with_mark(
            values=values_to_plot, mark_inds=(anom_plot_start, anom_plot_end), ax=ax
        )
        xticks = np.arange(start=0, stop=ind_max - ind_min, step=xticks_gran)
        xticklabels = xticks + ind_min
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_xlabel("timestamp", fontsize=12)
        ax.set_ylabel("value", fontsize=12)

    def anoms_lens(self) -> pd.Series:
        """Returns series in which i-th element contains
        length of the i-th anomaly
        """
        return self.anoms.apply(
            lambda x: x.loc["anom_end"] - x.loc["anom_start"], axis=1
        )

    @classmethod
    def load(cls, dir_path: str, name: str) -> TimeSeriesWithAnoms:
        """Loads time series object from files

        Args:
            dir_path: path to the directory with files
            name: name of the time series

        Returns:
            loaded TimeSeriesWithAnoms class object

        """
        values_filename, annotations_filename = cls.__get_filenames_for_name(name)

        # load values
        values = np.load(os.path.join(dir_path, values_filename))

        # load annotations (anoms)
        anoms = pd.read_feather(os.path.join(dir_path, annotations_filename))

        return TimeSeriesWithAnoms(values=values, anoms=anoms, name=name)

    @staticmethod
    def __check_values(values: np.ndarray):
        """Validates input values"""
        if not isinstance(values, np.ndarray):
            raise ValueError("values should be a numpy array")
        if isinstance(values, (pd.DataFrame, pd.Series)):
            raise ValueError("values should not be dataframe nor series")
        if len(values.shape) != 1:
            raise ValueError("values should be 1d array")

    @staticmethod
    def __check_anoms(anoms: list[tuple[int, int]] | pd.DataFrame):
        n_anoms = anoms.shape[0] if isinstance(anoms, pd.DataFrame) else len(anoms)
        if n_anoms == 0:
            raise ValueError("there must be at least one anomaly for a series")

    @staticmethod
    def __get_filenames_for_name(name: str):
        """For given time series name, returns tuple:
        (values_filename, annotations_filename)
        """
        values_filename = f"ts_{name}_values.npy"
        annotations_filename = f"ts_{name}_annot.feather"
        return values_filename, annotations_filename


class SeriesSet:
    """Represents a set of time series.

    Each time series must be treated separately in the experiment.
    What this class does is to provide a convenient container
    for a set of independent time series (e.g. in order to
    load/save them all at once)

    """

    def __init__(self, series_set: list[TimeSeriesWithAnoms], name: str):

        self.__check_series_set(series_set)
        self.series_set: list[TimeSeriesWithAnoms] = series_set
        self.name: str = name

    def save(self, dir_path: str):
        """Saves time series set into files so that it can be
        loaded later

        Args:
            dir_path: path to the directory where series set
                is to be saved; it must exist and be empty

        """

        # check if dir_path is an empty directory
        if len(os.listdir(dir_path)) != 0:
            raise RuntimeError("dir_path is not empty")

        # save dataset info
        series_set_info = {
            "name": self.name,
            "series_names": [ts.name for ts in self.series_set],
        }
        with open(os.path.join(dir_path, "series_set_info.json"), "w") as f:
            json.dump(series_set_info, f)

        # save series
        for ts in self.series_set:
            ts.save(dir_path)

    def get_series_dict(self) -> dict[str, TimeSeriesWithAnoms]:
        """Returns dictionary with series from series set:
        {series_name: series_object}
        """
        return {ts.name: ts for ts in self.series_set}

    @classmethod
    def load(cls, dir_path: str) -> SeriesSet:
        """Loads series set from files

        Args:
            dir_path: path to the directory with series set
                files

        Returns:
            loaded SeriesSet object

        """

        # load series set info
        with open(os.path.join(dir_path, "series_set_info.json")) as f:
            series_set_info = json.load(f)

        # load series set
        series_set = [
            TimeSeriesWithAnoms.load(dir_path=dir_path, name=name)
            for name in series_set_info["series_names"]
        ]

        return SeriesSet(series_set=series_set, name=series_set_info["name"])

    @staticmethod
    def __check_series_set(series_set: list[TimeSeriesWithAnoms]):
        """Validates given list of time series objects"""
        # check if not empty
        if len(series_set) == 0:
            raise ValueError("series_set must contain at least one series")
        # check if all series are TimeSeriesWithAnoms instances
        if any(not isinstance(ts, TimeSeriesWithAnoms) for ts in series_set):
            raise ValueError("all series must be TimeSeriesWithAnoms instances")
        # check if names are unique
        series_names = [ts.name for ts in series_set]
        if len(series_names) != len(set(series_names)):
            raise ValueError("All series names must be unique")


class TimeSeriesDescriptor(ABC):
    """Base class for descriptor classes. An instance of a descriptor
    class provides values of some features for a given time series
    object
    """

    @abstractmethod
    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:
        """Obtains values of time series-describing features for
        given time series object

        Args:
            ts: TimeSeriesWithAnoms instance representing time series
                to be described

        Returns:
            dict: keys are features and values are values of features
                that describe the time series

        """
        pass


class SeriesSetAnalyzer:
    """A tool to describe time series objects in
    a series set
    """

    def __init__(self, descriptors: list[TimeSeriesDescriptor]):
        self.descriptors: list[TimeSeriesDescriptor] = descriptors

    def __describe_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:
        """Gets description for a given time series object

        Args:
            ts (TimeSeriesWithAnoms): time series to be described

        Returns:
            dict: features and their values that describe the time series
                (keys - features, values - values of features for ts)
        """
        features_dicts = [dsc.get_value_for_ts(ts) for dsc in self.descriptors]
        return dict(sum([list(di.items()) for di in features_dicts], start=[]))

    def describe_series_set(self, series_set: SeriesSet) -> pd.DataFrame:
        """Describes time series in given series set

        Each time series is described using self.descriptors

        Args:
            series_set SeriesSet object

        Returns:
            pd.DataFrame: dataframe in which rows are indexed with names of
                time series and columns are names of features
        """
        series_descriptions = {
            ts.name: self.__describe_ts(ts)
            for ts in tqdm(series_set.series_set, desc=series_set.name)
        }
        return pd.DataFrame.from_dict(series_descriptions, orient="index")


class MapTransformation(ABC):
    """Base class for classes implementing map transformations,
    i.e. transformations that transform one series into another
    series

    Attributes:
        name: the name of the transformation performed

    """

    def __init__(self, name: str):
        self.name: str = name

    def transform(self, series_set: SeriesSet) -> SeriesSet:
        """Transforms each series in a series set and returns new
        series set

        Output series set contains transformed time series and its
        name is the name of the input series set with added name
        of the transformation

        Args:
            series_set: input SeriesSet class object, containing time series
                to apply transformation on

        Returns:
            SeriesSet: output object with time series transformed and
                name with transformation name added

        """
        transformed_set: list[TimeSeriesWithAnoms] = []
        for ts in tqdm(series_set.series_set, desc=self.name):
            try:
                transformed_set.append(self._transform_series(ts))
            except RejectSeries:
                warnings.warn(
                    f"Transforming series {ts.name} produced inconsistent output."
                    f" Omitting series."
                )
            except RuntimeError as e:
                print(f"An error occurred during transforming series {ts.name}")
                print(f"The series has been skipped. Error:")
                print(e)
        transformed_name = f"{series_set.name}_{self.name}"
        return SeriesSet(series_set=transformed_set, name=transformed_name)

    @abstractmethod
    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:
        """Gets one time series as an input and returns new time series
        which is the transformed version of the input

        This method should transform time series values and take care
        of correct anomaly annotation in the result series

        Args:
            ts: input time series object

        Returns:
            TimeSeriesWithAnoms: output time series object, with transformed
                values and anomalies annotations

        """
        pass


class SubsequenceAnomalyDetector(ABC):

    def __init__(self, name: str):
        self.name: str = name
        self.__fitted: bool = False

    def fit(self, X: np.ndarray):
        self.__check_X(X)
        self._fit(X)
        self.__fitted = True
        return self

    def get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        if not self.__fitted:
            raise RuntimeError("Detector must be fitted before finding anoms")
        k_anoms = self._get_k_anoms(k)
        self.__check_anoms(k_anoms, k)
        return k_anoms

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def _get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        pass

    @staticmethod
    def __check_X(X: np.ndarray) -> None:
        # check if X is 1D
        if len(X.shape) != 1:
            raise ValueError("X must be 1D array")
        # check if there is no NaN
        if np.any(np.isnan(X)):
            raise ValueError("X must not contain any NaNs")

    @staticmethod
    def __check_anoms(anoms: list[tuple[int, int]], k: int) -> None:
        # check if each anom is tuple of ints of length 2
        for anom in anoms:
            if (
                (not isinstance(anom, tuple))
                or (len(anom) != 2)
                or (not isinstance(anom[0], int))
                or (not isinstance(anom[1], int))
            ):
                raise RuntimeError(
                    f"Returned anomaly indices: {anom} are invalid. Each returned anomalous indices"
                    f"must be a tuple of ints of length 2"
                )
        # check if each anom_start is less than anom_end
        for anom_start, anom_end in anoms:
            if anom_start >= anom_end:
                raise RuntimeError(
                    f"Retuned anomaly indices: {anom_start, anom_end} are invalid. anom_start must be"
                    f"lower than anom_end"
                )
        # check if there are exactly k anoms
        if len(anoms) != k:
            raise RuntimeError(
                f"Anomaly detector should have returned k={k} anomalies, but"
                f"returned {len(anoms)}"
            )

        # check if anomalies do not overlap
        series_len_to_check = max([anom_end for _, anom_end in anoms])
        if are_anomalies_overlapping(anoms=anoms, series_len=series_len_to_check):
            raise RuntimeError(f"Anomalies must not overlap, but got overlapping")


class DetectionResultForSeries:

    def __init__(
        self,
        ts_name: str,
        method_name: str,
        detected_anoms: list[tuple[int, int]] | pd.DataFrame,
    ):
        self.ts_name = ts_name
        self.method_name: str = method_name
        self.detected_anoms: pd.DataFrame = (
            detected_anoms
            if isinstance(detected_anoms, pd.DataFrame)
            else pd.DataFrame(detected_anoms, columns=["anom_start", "anom_end"])
        )

    def save(self, dir_path: str) -> None:
        """Saves the object into file so that it
        can be loaded later

        Args:
            dir_path: path to the directory where to save file
                with detected anomalies indices

        """
        detected_anoms_filename = self.__get_filename_for_name(
            ts_name=self.ts_name, method_name=self.method_name
        )

        # save detected anoms
        self.detected_anoms.to_feather(os.path.join(dir_path, detected_anoms_filename))

    @classmethod
    def load(
        cls, dir_path: str, ts_name: str, method_name: str
    ) -> DetectionResultForSeries:
        """Loads detection result object from file

        Args:
            dir_path: path to the directory with files
            ts_name: name of the time series
            method_name: name of the method used to detection

        Returns:
            loaded TimeSeriesWithAnoms class object

        """
        detected_anoms_filename = cls.__get_filename_for_name(
            ts_name=ts_name, method_name=method_name
        )

        # load detected anoms df
        detected_anoms = pd.read_feather(
            os.path.join(dir_path, detected_anoms_filename)
        )

        return DetectionResultForSeries(
            ts_name=ts_name, method_name=method_name, detected_anoms=detected_anoms
        )

    @staticmethod
    def __get_filename_for_name(ts_name: str, method_name: str):
        """For given time series name, returns filename
        to save detected anomalies indices
        """
        detected_anoms_filename = f"ts_{ts_name}_result_{method_name}.feather"
        return detected_anoms_filename


class SeriesSetResult:

    def __init__(
        self,
        set_name: str,
        method_name: str,
        series_results: list[DetectionResultForSeries],
    ):
        self.set_name: str = set_name
        if any(ts_result.method_name != method_name for ts_result in series_results):
            raise ValueError(
                f"Inconsistent input - some series result have method name"
                f"different than passed method name"
            )
        self.method_name = method_name
        self.series_results: list[DetectionResultForSeries] = series_results

    def save(self, dir_path: str):
        """Saves results set into files so that it can be
        loaded later

        Args:
            dir_path: path to the directory where results
                are to be saved; it must exist and be empty

        """

        # check if dir_path is an empty directory
        if len(os.listdir(dir_path)) != 0:
            raise RuntimeError("dir_path is not empty")

        # save dataset info
        series_results_info = {
            "set_name": self.set_name,
            "method_name": self.method_name,
            "series_names": [ts_result.ts_name for ts_result in self.series_results],
        }
        with open(os.path.join(dir_path, "series_results_info.json"), "w") as f:
            json.dump(series_results_info, f)

        # save series
        for ts_result in self.series_results:
            ts_result.save(dir_path)

    def get_results_dict(self) -> dict[str, DetectionResultForSeries]:
        return {
            series_result.ts_name: series_result
            for series_result in self.series_results
        }

    @classmethod
    def load(cls, dir_path: str) -> SeriesSetResult:
        """Loads series set result from files

        Args:
            dir_path: path to the directory with series set result
                files

        Returns:
            loaded SeriesSetResult object

        """

        # load series set info
        with open(os.path.join(dir_path, "series_results_info.json")) as f:
            series_results_info = json.load(f)

        # load series set results
        series_results = [
            DetectionResultForSeries.load(
                dir_path=dir_path,
                ts_name=ts_name,
                method_name=series_results_info["method_name"],
            )
            for ts_name in series_results_info["series_names"]
        ]

        return SeriesSetResult(
            series_results=series_results,
            set_name=series_results_info["set_name"],
            method_name=series_results_info["method_name"],
        )
