from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class TrajectoryStyle:
    connection_style: str = "arc3,rad=0.0"
    arrow_style: str = "-|>"
    ordering: dict = field(default_factory=dict)
    merge_repeated_states: bool = True

    # todo :: why does this exist
    def add_ordering(self, axis, ordering):
        """ make new copy for ordering"""
        self.ordering[axis] = list(ordering)


# todo :: is this really the right data layout...? AOS vs SOA I guess
@dataclass
class ProcessedTrajData:
    valid: bool = False
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    t: list = field(default_factory=list)
    loops: set = field(default_factory=set)
    nodes: list = field(default_factory=list)
    offset_x: list = field(default_factory=list)
    offset_y: list = field(default_factory=list)
    bin_counts: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    data_x: list
    data_y: list
    data_t: list
    meta: dict = field(default_factory=dict)
    style: TrajectoryStyle = field(default_factory=TrajectoryStyle)
    id: int = None  # set in __post_init__

    # To cache processed data
    processed_data : ProcessedTrajData = field(default_factory=ProcessedTrajData)

    # static count of number of trajectories - use as a stand in for ID
    # todo :: unsure if this is daft. probably daft?
    next_id: ClassVar[int] = 1

    def __post_init__(self):
        self.id = self.next_id
        type(self).next_id += 1
        for data in [self.data_x, self.data_y]:
            if len(data) == len(self.data_t):
                data.pop(-1)  # truncate NaN data (????? todo :: ??)
    
    # Make it easier to add ordering to trajectory variables
    def add_x_ordering(self, ordering):
        self.style.add_ordering("x", ordering)
        
    # Make it easier to add ordering to trajectory variables
    def add_y_ordering(self, ordering):
        self.style.add_ordering("y", ordering)
        
    # Make it easier to add ordering to trajectory variables
    def add_global_ordering(self, ordering):
        self.style.add_ordering("x", ordering)
        self.style.add_ordering("y", ordering)

    def get_duration(self):
        return self.data_t[-1] - self.data_t[0]

    def get_num_visits(self):
        if self.style.merge_repeated_states:
            return len(self.processed_data.x)
        return 1 + sum(
            x0 != x1 or y0 != y1  # discount consecutive repeats
            for x0, x1, y0, y1 in zip(self.data_x, self.data_x[1:], self.data_y, self.data_y[1:])
        )

    def get_cell_range(self):
        # todo :: double check this does as intended
        return sum(map(len, self.processed_data.bin_counts.values()))

    def __merge_equal_adjacent_states(self):
        """
        Merge adjacent equal states and return merged data.
        Does not edit data within the trajectory as trajectories may contain >2(+time) variables
        """
        # todo :: bleh
        merge_count = 0
        for i in range(len(self.data_x)):
            if i != 0 and (self.data_x[i], self.data_y[i]) == (self.data_x[i - 1], self.data_y[i - 1]):
                merge_count += 1
                self.processed_data.loops.add(i - merge_count)
            else:
                self.processed_data.x.append(self.data_x[i])
                self.processed_data.y.append(self.data_y[i])
                self.processed_data.t.append(self.data_t[i])
        self.processed_data.t.append(self.data_t[-1])
        if "x" in self.style.ordering:
            self.processed_data.x = convert_on_ordering(self.processed_data.x, self.style.ordering["x"])
        if "y" in self.style.ordering:
            self.processed_data.y = convert_on_ordering(self.processed_data.y, self.style.ordering["y"])

    def process_data(self) -> bool:
        """
        processes data(? you'd hope)
        returns whether data has already been processed
        """
        # check if already processed
        if self.processed_data.valid:
            return False
        if self.style.merge_repeated_states:
            self.__merge_equal_adjacent_states()
        else:
            self.processed_data.t = self.data_t
            self.processed_data.x = self.data_x
            self.processed_data.y = self.data_y

        for x, y in zip(self.processed_data.x, self.processed_data.y):
            # todo :: defaultdict / Counter
            if y in self.processed_data.bin_counts:
                self.processed_data.bin_counts[y][x] = self.processed_data.bin_counts[y].get(x, 0) + 1
            else:
                self.processed_data.bin_counts[y] = {x: 1}

        self.processed_data.nodes = [
            latter - former
            for former, latter in zip(self.processed_data.t, self.processed_data.t[1:])
        ]
        self.processed_data.valid = True
        return True


def convert_on_ordering(data, ordering):
    index = {ordering[i] : i for i in range(len(ordering))}
    return [index[x] for x in data]