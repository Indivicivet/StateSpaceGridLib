import csv
import warnings
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import ClassVar, Optional, Tuple

import networkx as nx
import numpy as np


@dataclass
class Trajectory:
    data_x: list
    data_y: list
    data_t: list
    # todo :: data_t (onsets) should be replaced by durations

    id: int = None  # set in __post_init__
    # static count of number of trajectories - use as a stand in for ID
    # todo :: unsure if this is daft. probably daft?
    next_id: ClassVar[int] = 1

    def __post_init__(self):
        if self.id is None:
            self.id = self.next_id
        type(self).next_id += 1
        # todo :: removed some "pop NaNs from the end" code here
        assert (
            len(self.data_x) == (len(self.data_t) - 1)
            and len(self.data_y) == (len(self.data_t) - 1)
        ), (
            "Time data should be of length 1 longer than x and y data"
            f", but got lengths {len(self.data_x)=} {len(self.data_y)=} {len(self.data_t)=}"
        )

    # return number of cell transitions plus 1
    def get_num_visits(self) -> int:
        return 1 + sum(
            (x, y) != (x1, y1)
            for x, x1, y, y1 in zip(self.data_x, self.data_x[1:], self.data_y, self.data_y[1:])
        )

    # return number of unique cells visited
    def get_cell_range(self) -> int:
        return len(set(zip(self.data_x, self.data_y)))

    # return formatted state data
    def get_states(
        self,
        x_ordering: Optional[list] = None,
        y_ordering: Optional[list] = None,
        merge_repeated_states: bool = True,
    ) -> Tuple[list, list, list, set]:
        # todo :: can we deal with reordering at the call site?
        def maybe_reorder(data, ordering=None):
            if not ordering:
                return data
            return [ordering.index(x) for x in data]

        if not merge_repeated_states:
            return (
                maybe_reorder(self.data_x, x_ordering),
                maybe_reorder(self.data_y, y_ordering),
                self.data_t,
                set(),
            )

        x_merged = [self.data_x[0]]
        y_merged = [self.data_y[0]]
        t_merged = [self.data_t[0]]
        loops = set()
        for x, x_1, y, y_1, t in zip(
                self.data_x,
                self.data_x[1:],
                self.data_y,
                self.data_y[1:],
                self.data_t[1:],
        ):
            if (x, y) == (x_1, y_1):
                loops.add(len(x_merged) - 1)
            else:
                x_merged.append(x_1)
                y_merged.append(y_1)
                t_merged.append(t)
        t_merged.append(self.data_t[-1])
        return (
            maybe_reorder(x_merged, x_ordering),
            maybe_reorder(y_merged, y_ordering),
            t_merged,
            loops,
        )

    def calculate_dispersion(
        self,
        total_cells: int,
    ) -> float:
        cell_durations = Counter()
        for x, y, t1, t2 in zip(
            self.data_x,
            self.data_y,
            self.data_t,
            self.data_t[1:],
        ):
            cell_durations[(x, y)] += t2 - t1
        return 1 - (
            (
                total_cells * sum(x ** 2 for x in cell_durations.values())
                / sum(cell_durations.values()) ** 2
            )
            - 1
        ) / (total_cells - 1)

    # construct trajectory from legacy trj file
    @classmethod
    def from_legacy_trj(
        cls,
        filename,
        params=(1, 2),
    ):
        """For legacy .trj files. Stay away, they're ew!"""
        warnings.warn(
            "This is just provided for testing against specific"
            " legacy behaviour. trj files are ew, be careful!"
        )
        onset = []
        v1 = []
        v2 = []
        with open(filename) as f:
            for line in csv.reader(f, delimiter="\t"):
                if line[0].lower() == "onset":
                    continue
                if len(line) == 1:
                    onset.append(float(line[0]))
                    break
                elif len(line) < 3:
                    break
                onset.append(float(line[0]))
                v1.append(int(line[params[0]]))
                v2.append(int(line[params[1]]))
        return cls(v1, v2, onset)
