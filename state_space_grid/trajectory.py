import csv
import warnings
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import ClassVar, Optional


@dataclass
class TrajectoryStyle:
    connection_style: str = "arc3,rad=0.0"
    arrow_style: str = "-|>"
    merge_repeated_states: bool = True


@dataclass
class Trajectory:
    data_x: list
    data_y: list
    data_t: list
    # todo :: data_t (onsets) should be replaced by durations
    meta: dict = field(default_factory=dict)
    style: TrajectoryStyle = field(default_factory=TrajectoryStyle)
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
        x_ordering: Optional[list]= None,
        y_ordering: Optional[list] = None
    ) -> tuple[list, list, list, set]:
        if self.style.merge_repeated_states:

            def merge_equal_adjacent_states(
                    x_data,
                    y_data,
                    t_data,
                    # todo :: this x_ordering, y_ordering is used in multiple places
                    # consider refactor
                    x_ordering=None,
                    y_ordering=None,
            ):
                """
                Merge adjacent equal states and return merged data.
                Does not edit data within the trajectory as trajectories may
                contain >2(+time) variables
                """
                x_merged = [x_data[0]]
                y_merged = [y_data[0]]
                t_merged = [t_data[0]]
                loops = set()
                for x, x_1, y, y_1, t in zip(
                        x_data,
                        x_data[1:],
                        y_data,
                        y_data[1:],
                        t_data[1:],
                ):
                    if (x, y) == (x_1, y_1):
                        loops.add(len(x_merged) - 1)
                    else:
                        x_merged.append(x_1)
                        y_merged.append(y_1)
                        t_merged.append(t)
                t_merged.append(t_data[-1])
                if x_ordering:
                    x_merged = [x_ordering.index(val) for val in x_merged]
                if y_ordering:
                    y_merged = [y_ordering.index(val) for val in y_merged]
                return x_merged, y_merged, t_merged, loops

            return merge_equal_adjacent_states(
                self.data_x,
                self.data_y,
                self.data_t,
                x_ordering,
                y_ordering,
            )

        def maybe_reorder(data, ordering=None):
            if not ordering:
                return data
            return [ordering.index(x) for x in data]

        return (
            maybe_reorder(self.data_x, x_ordering),
            maybe_reorder(self.data_y, y_ordering),
            self.data_t,
            set(),
        )

    def calculate_dispersion(
        self,
        total_cells: int,
    ) -> float:
        cell_durations = Counter(
            t2 - t1
            for x, y, t1, t2 in zip(
                self.data_x,
                self.data_y,
                self.data_t,
                self.data_t[1:],
            )
        )
        return 1 - (
            (
                total_cells * sum(x ** 2 for x in cell_durations.values())
                / cell_durations.total() ** 2
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
