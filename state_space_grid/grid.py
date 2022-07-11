from dataclasses import dataclass
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import numpy as np
from statistics import mean
import fractions
from .trajectory import Trajectory, TrajectoryStyle, check_trajectory_list, ProcessedTrajData
from .states import *


@dataclass
class GridStyle:
    title: str = ""
    label_font_size: int = 14
    tick_font_size: int = 14
    title_font_size: int = 14
    tick_increment_x: Optional[int] = None
    tick_increment_x: Optional[int] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_min: Optional[int] = None  # todo :: are they floats?
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    rotate_x_labels: bool = False

    @property
    def x_minmax_given(self):
        return self.x_min is not None, self.x_max is not None

    @property
    def y_minmax_given(self):
        return self.y_min is not None, self.y_max is not None


class GridCumulativeData:
    def __init__(self):
        self.valid = False
        self.max_duration = 0
        self.rounded_x_min = 0
        self.rounded_x_max = 0
        self.rounded_y_min = 0
        self.rounded_y_max = 0
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
        self.bin_counts = dict()
        self.cell_size_x = 0
        self.cell_size_y = 0

    def clear(self):
        self.valid = False
        self.max_duration = 0
        self.rounded_x_min = 0
        self.rounded_x_max = 0
        self.rounded_y_min = 0
        self.rounded_y_max = 0
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
        self.bin_counts = []
        self.cell_size_x = 0
        self.cell_size_y = 0


class GridMeasures:
    def __init__(self):
        self.trajectory_ids = []
        self.mean_duration = 0
        self.mean_number_of_events = 0
        self.mean_number_of_visits = 0
        self.mean_cell_range = 0
        self.overall_cell_range = 0
        self.mean_duration_per_event = 0
        self.mean_duration_per_visit = 0
        self.mean_duration_per_cell = 0
        self.dispersion = 0
        self.mean_missing_events = 0
        self.mean_missing_duration = 0

    def __str__(self):
        return "\n".join(["trajectory ids: {}".format(", ".join([str(x) for x in self.trajectory_ids])),
                          "mean duration: {}".format(self.mean_duration),
                          "mean number of events: {}".format(self.mean_number_of_events),
                          "mean number of visits: {}".format(self.mean_number_of_visits),
                          "mean cell range: {}".format(self.mean_cell_range),
                          "overall cell range: {}".format(self.overall_cell_range),
                          "mean duration per event: {}".format(self.mean_duration_per_event),
                          "mean duration per visit: {}".format(self.mean_duration_per_visit),
                          "mean duration per cell: {}".format(self.mean_duration_per_cell),
                          "dispersion {}".format(self.dispersion),
                          "mean missing events {}".format(self.mean_missing_events),
                          "mean missing duration {}".format(self.mean_missing_duration)])


class Grid:
    def __init__(self, trajectories, style=GridStyle()):
        self.trajectory_list = [i for i in trajectories]
        self.graph = nx.Graph()
        self.ax = plt.gca()
        self.style = style
        self._processed_data = GridCumulativeData()
        check_trajectory_list(self.trajectory_list)

    def __set_background(self, x_min, y_min, x_scale, y_scale, x_max, y_max):
        background_colours = ListedColormap([np.array([220 / 256, 220 / 256, 220 / 256, 1]), np.array([1, 1, 1, 1])])
        background = [[((i + j) % 2) for i in range(int(x_min / x_scale), int(x_max / x_scale) + 1)] for j in
                      range(int(y_min / y_scale), int(y_max / y_scale) + 1)]
        self.ax.imshow(background,
                       extent=[int(x_min) - 0.5 * x_scale, int(x_max) + 0.5 * x_scale, int(y_min) - 0.5 * y_scale,
                               int(y_max) + 0.5 * y_scale], cmap=background_colours, interpolation='none')

    def __draw_graph(self, trajectory):
        # Create a dictionary to define positions for node numbers
        num_nodes = len(trajectory.processed_data.x)
        pos = {i: (trajectory.processed_data.x[i], trajectory.processed_data.y[i]) for i in range(num_nodes)}

        # List of tuples to define edges between nodes
        edges = [(i, i + 1) for i in range(num_nodes - 1)]
        for loop_node in trajectory.processed_data.loops:
            edges.append((loop_node, loop_node))

        # Calculate node size based on node data
        node_size_scale_factor = 1000 / self._processed_data.max_duration
        node_sizes = list(map(lambda n: n * node_size_scale_factor, trajectory.processed_data.nodes))

        # Add nodes and edges to graph
        self.graph.add_nodes_from(pos.keys())
        self.graph.add_edges_from(edges)

        # Draw graphs
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color='indigo')
        nx.draw_networkx_edges(self.graph, pos, node_size=node_sizes, nodelist=[i for i in range(num_nodes)],
                               edgelist=edges, arrows=True, arrowstyle=trajectory.style.arrow_style, node_shape='.',
                               arrowsize=10, width=2, connectionstyle=trajectory.style.connection_style, )

    def __draw_ticks(self, drawstyle):
        # all of this needs to go in a separate function, called with show()
        # we need to store which axis is which column - kick up a fuss if future plots don't match this
        # we also need to store an idea of minimum scale - this is going to fuck us if scales don't match
        # - maybe just tell people if they have scale adjustment problems to specify their own

        # Get tick labels - either numeric or categories
        if self.style.x_order:
            tick_label_x = self.style.x_order
        else:
            tick_label_x = drawstyle.ordering["x"] if "x" in drawstyle.ordering else [str(i) for i in
                                                                                      range(
                                                                                          self._processed_data.rounded_x_min,
                                                                                          self._processed_data.rounded_x_max + 1,
                                                                                          self._processed_data.cell_size_x)]
        if self.style.y_order:
            tick_label_y = self.style.y_order
        else:
            tick_label_y = drawstyle.ordering["y"] if "y" in drawstyle.ordering else [str(i) for i in
                                                                                      range(
                                                                                          self._processed_data.rounded_y_min,
                                                                                          self._processed_data.rounded_y_max + 1,
                                                                                          self._processed_data.cell_size_y)]
        # Set ticks for states
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.tick_params(axis='x', labelsize=self.style.tickfontsize,
                            rotation=90 if self.style.rotate_xlabels else 0)
        self.ax.tick_params(axis='y', labelsize=self.style.tickfontsize)
        self.ax.xaxis.set_major_locator(ticker.FixedLocator([i for i in range(self._processed_data.rounded_x_min,
                                                                              self._processed_data.rounded_x_max + 1,
                                                                              self._processed_data.cell_size_x)]))
        self.ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_label_x))
        self.ax.yaxis.set_major_locator(ticker.FixedLocator([i for i in range(self._processed_data.rounded_y_min,
                                                                              self._processed_data.rounded_y_max + 1,
                                                                              self._processed_data.cell_size_y)]))
        self.ax.yaxis.set_major_formatter(ticker.FixedFormatter(tick_label_y))

        # Set axis labels
        self.ax.set_xlabel(self.style.x_label, fontsize=self.style.label_font_size)
        self.ax.set_ylabel(self.style.y_label, fontsize=self.style.label_font_size)

        if self.style.title:
            self.ax.set_title(self.style.title, fontsize=self.style.title_font_size)

    def __offset_trajectories(self):
        # get total bin counts
        for trajectory in self.trajectory_list:
            for y, x_and_count in trajectory.processed_data.bin_counts.items():
                for x, count in x_and_count.items():
                    if y in self._processed_data.bin_counts:
                        self._processed_data.bin_counts[y][x] = self._processed_data.bin_counts[y].get(x, 0) + count
                    else:
                        self._processed_data.bin_counts[y] = {x: count}
        current_bin_counter = {y: {x: 0 for x in v.keys()} for y, v in self._processed_data.bin_counts.items()}
        for trajectory in self.trajectory_list:
            # If same state is repeated, offset states so they don't sit on top of one another:
            offset_within_bin(trajectory.processed_data.x, self._processed_data.cell_size_x,
                              trajectory.processed_data.y, self._processed_data.cell_size_y,
                              self._processed_data.bin_counts, current_bin_counter)

    def __draw_background_and_view(self):
        # Make an estimate for scale size of checkerboard grid sizing
        self._processed_data.cell_size_x = calculate_scale(self._processed_data.x_min,
                                                           self._processed_data.x_max) if self.style.tick_increment_x is None else self.style.tick_increment_x
        self._processed_data.cell_size_y = calculate_scale(self._processed_data.y_min,
                                                           self._processed_data.y_max) if self.style.tick_increment_y is None else self.style.tick_increment_y

        self._processed_data.rounded_x_min = self._processed_data.x_min - (
                    self._processed_data.x_min % self._processed_data.cell_size_x)
        self._processed_data.rounded_x_max = self._processed_data.x_max + (self._processed_data.cell_size_x - ((
                                                                                                                           self._processed_data.x_max % self._processed_data.cell_size_x) if self._processed_data.x_max % self._processed_data.cell_size_x else self._processed_data.cell_size_x))
        self._processed_data.rounded_y_min = self._processed_data.y_min - (
                    self._processed_data.y_min % self._processed_data.cell_size_y)
        self._processed_data.rounded_y_max = self._processed_data.y_max + (self._processed_data.cell_size_y - ((
                                                                                                                           self._processed_data.y_max % self._processed_data.cell_size_y) if self._processed_data.y_max % self._processed_data.cell_size_y else self._processed_data.cell_size_y))

        x_padding = self._processed_data.cell_size_x / 2
        y_padding = self._processed_data.cell_size_y / 2

        # Set view of axes
        self.ax.set_xlim(
            [self._processed_data.rounded_x_min - x_padding, self._processed_data.rounded_x_max + x_padding])
        self.ax.set_ylim(
            [self._processed_data.rounded_y_min - y_padding, self._processed_data.rounded_y_max + y_padding])

        # Set background checkerboard:
        self.__set_background(self._processed_data.rounded_x_min, self._processed_data.rounded_y_min,
                              self._processed_data.cell_size_x, self._processed_data.cell_size_y,
                              self._processed_data.rounded_x_max, self._processed_data.rounded_y_max)

    def __process(self, trajectory):
        # Get relevant data (and do merging of repeated states if desired)

        if not trajectory.process_data():
            return  # early return for cases where trajectory data already processed

        # used for deciding node sizes
        self._processed_data.max_duration = max(max(trajectory.processed_data.nodes), self._processed_data.max_duration)

        # Get min and max values
        x_min, x_max = calculate_min_max(trajectory.processed_data.x)
        y_min, y_max = calculate_min_max(trajectory.processed_data.y)

        if self.style.x_minmax_given[0]:
            x_min = self.style.x_min
            if self._processed_data.x_min is None:
                self._processed_data.x_min = x_min
        else:
            # if x_min not given, we should keep track of this
            x_min = x_min if self._processed_data.x_min is None else min(x_min, self._processed_data.x_min)
            self._processed_data.x_min = x_min
        if self.style.x_minmax_given[1]:
            x_max = self.style.x_max
            if self._processed_data.x_max is None:
                self._processed_data.x_max = x_max
        else:
            x_max = x_max if self._processed_data.x_max is None else max(x_max, self._processed_data.x_max)
            self._processed_data.x_max = x_max
        if self.style.y_minmax_given[0]:
            y_min = self.style.y_min
            if self._processed_data.y_min is None:
                self._processed_data.y_min = y_min
        else:
            # if y_min not given, we should keep track of this
            y_min = y_min if self._processed_data.y_min is None else min(y_min, self._processed_data.y_min)
            self._processed_data.y_min = y_min
        if self.style.y_minmax_given[1]:
            y_max = self.style.y_max
            if self._processed_data.y_max is None:
                self._processed_data.y_max = y_max
        else:
            y_max = y_max if self._processed_data.y_max is None else max(y_max, self._processed_data.y_max)
            self._processed_data.y_max = y_max

    def set_style(self, gridstyle: GridStyle):
        self._processed_data.clear()
        self.style = gridstyle

    def get_style(self):
        return self.style

    def add_trajectory_data(self, *trajectories: Trajectory):
        for trajectory in trajectories:
            self.trajectory_list[trajectory.meta["ID"]] = trajectory
        if trajectories:
            self._processed_data.clear()
            check_trajectory_list(self.trajectory_list)

    def draw(self, save_as=""):
        if not self._processed_data.valid:
            for trajectory in self.trajectory_list:
                trajectory.processed_data.valid = False
                self.__process(trajectory)
        self.__draw_background_and_view()
        self.__offset_trajectories()
        for trajectory in self.trajectory_list:
            self.__draw_graph(trajectory)
        self.__draw_ticks(self.trajectory_list[0].style)
        self.ax.set_aspect('auto')
        plt.tight_layout()
        if save_as:
            self.ax.savefig(save_as)
        else:
            plt.show()
        self._processed_data.valid = True

    def __calculate_dispersion(self):
        num_cells_x = int((self._processed_data.x_max - self._processed_data.x_min) / self._processed_data.cell_size_x) + 1
        num_cells_y = int((self._processed_data.y_max - self._processed_data.y_min) / self._processed_data.cell_size_y) + 1
        n = num_cells_x * num_cells_y
        cell_durations = dict()
        total_duration = 0
        for traj in self.trajectory_list:
            for i in range(len(traj.processed_data.nodes)):
                y = traj.processed_data.y[i]
                x = traj.processed_data.x[i]
                duration = traj.processed_data.nodes[i]
                if y in cell_durations:
                    cell_durations[y][x] = cell_durations[y].get(x, 0) + duration
                else:
                    cell_durations[y] = {x: duration}
                total_duration += duration
        sum_d_D = sum([pow(fractions.Fraction(d, total_duration),2) for x_and_d in cell_durations.values() for d in x_and_d.values()])
        return fractions.Fraction(n * sum_d_D - 1, n - 1)

    def get_measures(self):
        if not self._processed_data.valid:
            # necessary processing
            for trajectory in self.trajectory_list:
                trajectory.processed_data.valid = False
                self.__process(trajectory)
            self.__draw_background_and_view()
            self.__offset_trajectories()
            self._processed_data.valid = True

        ids, durations, event_numbers, visit_numbers, \
            cell_ranges, dispersions, missing_events, missing_duration = [], [], [], [], [], [], [], []
        measures = GridMeasures()

        for traj in self.trajectory_list:
            measures.trajectory_ids.append(traj.meta["ID"])
            durations.append(traj.get_duration())
            event_numbers.append(len(traj.data_x))
            visit_numbers.append(traj.get_num_visits())
            cell_ranges.append(traj.get_cell_range())

        measures.mean_duration = mean(durations)
        measures.mean_number_of_events = mean(event_numbers)
        measures.mean_number_of_visits = mean(visit_numbers)
        measures.mean_cell_range = mean(cell_ranges)
        measures.overall_cell_range = len(
            [1 for x_and_count in self._processed_data.bin_counts.items() for x in x_and_count])
        measures.mean_duration_per_event = mean(map(lambda x, y: x / y, durations, event_numbers))
        measures.mean_duration_per_visit = mean(map(lambda x, y: x / y, durations, visit_numbers))
        measures.mean_duration_per_cell = mean(map(lambda x, y: x / y, durations, cell_ranges))
        measures.dispersion = float(self.__calculate_dispersion())
        return measures
