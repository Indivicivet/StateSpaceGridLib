class TrajectoryStyle:
    def __init__(self, connection_style="arc3,rad=0.0", arrow_style='-|>', ordering = dict(), merge_repeated_states=True):
        self.connection_style = connection_style
        self.arrow_style = arrow_style
        self.ordering={k:[x for x in v] for k,v in ordering.items()}
        self.merge_repeated_states = merge_repeated_states

    def add_ordering(self, axis, ordering):
        # make new copy for ordering
        self.ordering[axis] = [i for i in ordering]


class ProcessedTrajData:
    def __init__(self):
        valid = False
        self.x = []
        self.y = []
        self.t = []
        self.loops = set()
        self.nodes = [] # (durations)
        self.offset_x = []
        self.offset_y = []
        self.bin_counts = dict()


class Trajectory:
    numTrajectories = 0  # static count of number of trajectories - use as a stand in for ID

    def __init__(self, data_x, data_y, data_t, meta=None, style=TrajectoryStyle()):
        # todo :: remove nonsense
        self.numTrajectories = self.numTrajectories+1
        self.data_x = [x for x in data_x]
        self.data_y = [y for y in data_y]
        self.data_t = [t for t in data_t]
        self.meta = meta if meta else {}
        self.id = self.numTrajectories  # ???
        self.style = style
        truncate_nan_data(self.data_x, self.data_y, self.data_t)
        self.processed_data = ProcessedTrajData() # To cache processed data
    
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
        else:
            repeat_count = 0
            for i in range(1,len(self.data_x)):
                if (self.data_x[i], self.data_y[i]) == (self.data_x[i - 1], self.data_y[i - 1]):
                    repeat_count += 1
            return len(self.data_x) - repeat_count

    def get_cell_range(self):
        return len([1 for x_and_count in self.processed_data.bin_counts.items() for x in x_and_count])



    # Merge adjacent equal states and return merged data.
    # Does not edit data within the trajectory as trajectories may contain >2(+time) variables
    def __merge_equal_adjacent_states(self):
        merge_count = 0
        for i in range(len(self.data_x)):
            if i != 0 and (self.data_x[i], self.data_y[i]) == (self.data_x[i - 1], self.data_y[i - 1]):
                merge_count = merge_count + 1
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

    # returns if data has already been processed
    def process_data(self):
        # check if already processed
        if self.processed_data.valid:
            return False
        if self.style.merge_repeated_states:
            self.__merge_equal_adjacent_states()
        else:
            self.processed_data.t = self.data_t
            self.processed_data.x = self.data_x
            self.processed_data.y = self.data_y

        for i in range(len(self.processed_data.x)):
            y = self.processed_data.y[i]
            x = self.processed_data.x[i]
            if y in self.processed_data.bin_counts:
                self.processed_data.bin_counts[y][x] = self.processed_data.bin_counts[y].get(x, 0) + 1
            else:
                self.processed_data.bin_counts[y] = {x: 1}

        self.processed_data.nodes = [(self.processed_data.t[i + 1] - self.processed_data.t[i]) for i in range(len(self.processed_data.t) - 1)]
        self.processed_data.valid = True
        return True


def truncate_nan_data(x_data, y_data, t_data):
    nan_present_x = len(x_data) == len(t_data)
    nan_present_y = len(y_data) == len(t_data)
    if nan_present_x:
        x_data.pop(-1)
    if nan_present_y:
        y_data.pop(-1)


def check_trajectory_list(trajectory_list):
    assert trajectory_list, "A list of trajectories must be supplied to Grid"
    for trajectory in trajectory_list:
        assert isinstance(trajectory, Trajectory), "All trajectory data supplied in trajectory list must be instances of the trajectory class"

    meta_names = {column for column in trajectory_list[0].meta.keys()}
    for trajectory in trajectory_list:
        # possibly go for a more type-agnostic approach here? (see above)
        assert len(meta_names) == len \
            (trajectory.meta), "trajectory ID {}: metadata fields don't match with others in list".format(trajectory.id)
        for i in trajectory.meta.keys():
            assert i in meta_names, "metadata field name {} in trajectory ID {} does not exist in first trajectory".format(i, trajectory.id)


def convert_on_ordering(data, ordering):
    index = {ordering[i] : i for i in range(len(ordering))}
    return [index[x] for x in data]