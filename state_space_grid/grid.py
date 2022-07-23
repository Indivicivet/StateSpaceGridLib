from dataclasses import dataclass, field
from typing import Optional, Union
from numbers import Number

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import numpy as np
from statistics import mean
import fractions

from .trajectory import Trajectory
from . import util


@dataclass
class GridStyle:
    title: str = ""
    label_font_size: int = 14
    tick_font_size: int = 14
    title_font_size: int = 14
    tick_increment_x: Optional[int] = None
    tick_increment_y: Optional[int] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_min: Optional[int] = None  # todo :: are they floats?
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    x_order: list[Union[str, Number]] = field(default_factory=list)
    y_order: list[Union[str, Number]] = field(default_factory=list)
    rotate_x_labels: bool = False

    @property
    def x_minmax_given(self):
        return self.x_min is not None, self.x_max is not None

    @property
    def y_minmax_given(self):
        return self.y_min is not None, self.y_max is not None


@dataclass
class GridCumulativeData:
    valid: bool = False
    max_duration: int = 0
    rounded_x_min: int = 0
    rounded_x_max: int = 0
    rounded_y_min: int = 0
    rounded_y_max: int = 0
    x_min: Optional[int] = None
    y_min: Optional[int] = None
    x_max: Optional[int] = None
    y_max: Optional[int] = None
    bin_counts: dict = field(default_factory=lambda: {})
    cell_size_x: int = 0
    cell_size_y: int = 0


@dataclass
class GridMeasures:
    trajectory_ids: list = field(default_factory=lambda: [])
    mean_duration: float = 0
    mean_number_of_events: float = 0  # todo float ye?
    mean_number_of_visits: float = 0
    mean_cell_range: float = 0
    overall_cell_range: float = 0
    mean_duration_per_event: float = 0
    mean_duration_per_visit: float = 0
    mean_duration_per_cell: float = 0
    dispersion: float = 0
    mean_missing_events: float = 0
    mean_missing_duration: float = 0


class Grid:
    def __init__(self, trajectories, style=None):
        self.trajectory_list = list(trajectories)  # hm?
        self.graph = nx.Graph()
        self.ax = plt.gca()
        self.style = style if style is not None else GridStyle()
        self._processed_data = GridCumulativeData()

    def __set_background(self, x_min, y_min, x_scale, y_scale, x_max, y_max):
        jj, ii = np.mgrid[
            int(y_min / y_scale):int(y_max / y_scale) + 1,
            int(x_min / x_scale):int(x_max / x_scale) + 1
        ]
        self.ax.imshow(
            # checkerboard
            # todo :: maybe i messed it up
            (jj + ii) % 2,
            extent=[
                int(x_min) - 0.5 * x_scale,
                int(x_max) + 0.5 * x_scale,
                int(y_min) - 0.5 * y_scale,
                int(y_max) + 0.5 * y_scale
            ],
            cmap=ListedColormap([
                [220 / 256, 220 / 256, 220 / 256, 1],
                [1, 1, 1, 1],
            ]),
            interpolation='none',
        )

    def __draw_graph(self, trajectory):
        node_number_positions = dict(enumerate(zip(trajectory.processed_data.x, trajectory.processed_data.y)))

        # List of tuples to define edges between nodes
        edges = (
            [(i, i + 1) for i in range(len(trajectory.processed_data.x) - 1)]
            + [(loop_node, loop_node) for loop_node in trajectory.processed_data.loops]
        )

        node_sizes = list(  # todo :: maybe redundant list()
            (1000 / self._processed_data.max_duration)
            * np.array(trajectory.processed_data.nodes)
        )

        # Add nodes and edges to graph
        self.graph.add_nodes_from(node_number_positions.keys())
        self.graph.add_edges_from(edges)

        # Draw graphs
        nx.draw_networkx_nodes(self.graph, node_number_positions, node_size=node_sizes, node_color='indigo')
        nx.draw_networkx_edges(
            self.graph,
            node_number_positions,
            node_size=node_sizes,
            nodelist=list(range(len(trajectory.processed_data.x))),
            edgelist=edges,
            arrows=True,
            arrowstyle=trajectory.style.arrow_style,
            node_shape='.',
            arrowsize=10,
            width=2,
            connectionstyle=trajectory.style.connection_style,
        )

    def __draw_ticks(self, draw_style):
        # all of this needs to go in a separate function, called with show()
        # we need to store which axis is which column - kick up a fuss if future plots don't match this
        # we also need to store an idea of minimum scale - this is going to fuck us if scales don't match
        # - maybe just tell people if they have scale adjustment problems to specify their own

        # Get tick labels - either numeric or categories
        # todo :: this bit is a bit of a mess
        rounded_x_points = range(
            int(self._processed_data.rounded_x_min),
            int(self._processed_data.rounded_x_max + 1),
            int(self._processed_data.cell_size_x),
        )
        rounded_y_points = range(
            int(self._processed_data.rounded_y_min),
            int(self._processed_data.rounded_y_max + 1),
            int(self._processed_data.cell_size_y),
        )
        tick_label_x = (
            self.style.x_order
            or draw_style.ordering.get("x", [str(i) for i in rounded_x_points])
        )
        tick_label_y = (
            self.style.y_order
            or draw_style.ordering.get("y", [str(i) for i in rounded_y_points])
        )
        # Set ticks for states
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.tick_params(
            axis='x',
            labelsize=self.style.tick_font_size,
            rotation=90 * self.style.rotate_x_labels,
        )
        self.ax.tick_params(
            axis='y',
            labelsize=self.style.tick_font_size,
        )
        self.ax.xaxis.set_major_locator(ticker.FixedLocator(rounded_x_points))
        self.ax.yaxis.set_major_locator(ticker.FixedLocator(rounded_y_points))
        self.ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_label_x))
        self.ax.yaxis.set_major_formatter(ticker.FixedFormatter(tick_label_y))

        # Set axis labels
        self.ax.set_xlabel(self.style.x_label, fontsize=self.style.label_font_size)
        self.ax.set_ylabel(self.style.y_label, fontsize=self.style.label_font_size)

        if self.style.title:
            self.ax.set_title(self.style.title, fontsize=self.style.title_font_size)

    def __offset_trajectories(self):
        # todo :: later -- need to think about what this actually does
        # get total bin counts
        for trajectory in self.trajectory_list:
            for y, x_and_count in trajectory.processed_data.bin_counts.items():
                for x, count in x_and_count.items():
                    # todo :: defaultdict/Counter, and the (y, x) thing
                    if y in self._processed_data.bin_counts:
                        self._processed_data.bin_counts[y][x] = self._processed_data.bin_counts[y].get(x, 0) + count
                    else:
                        self._processed_data.bin_counts[y] = {x: count}
        current_bin_counter = {y: {x: 0 for x in v.keys()} for y, v in self._processed_data.bin_counts.items()}
        for trajectory in self.trajectory_list:
            # If same state is repeated, offset states so they don't sit on top of one another:
            util.offset_within_bin(
                trajectory.processed_data.x,
                self._processed_data.cell_size_x,
                trajectory.processed_data.y,
                self._processed_data.cell_size_y,
                self._processed_data.bin_counts,
                current_bin_counter,
            )

    def __draw_background_and_view(self):
        # todo :: whole bunch of stuff that is a bit messy here
        # Make an estimate for scale size of checkerboard grid sizing
        self._processed_data.cell_size_x = (
            util.calculate_scale(self._processed_data.x_max - self._processed_data.x_min)
            if self.style.tick_increment_x is None
            else self.style.tick_increment_x
        )
        self._processed_data.cell_size_y = (
            util.calculate_scale(self._processed_data.y_max - self._processed_data.y_min)
            if self.style.tick_increment_y is None
            else self.style.tick_increment_y
        )

        self._processed_data.rounded_x_min = (
            self._processed_data.x_min
            - (self._processed_data.x_min % self._processed_data.cell_size_x)
        )
        self._processed_data.rounded_x_max = (
            self._processed_data.x_max
            + self._processed_data.cell_size_x
            - (
               self._processed_data.x_max % self._processed_data.cell_size_x
               if self._processed_data.x_max % self._processed_data.cell_size_x
               else self._processed_data.cell_size_x
            )
        )
        self._processed_data.rounded_y_min = (
            self._processed_data.y_min
            - (self._processed_data.y_min % self._processed_data.cell_size_y)
        )
        self._processed_data.rounded_y_max = (
            self._processed_data.y_max
            + self._processed_data.cell_size_y
            - (
                self._processed_data.y_max % self._processed_data.cell_size_y
                if self._processed_data.y_max % self._processed_data.cell_size_y
                else self._processed_data.cell_size_y
            )
        )

        x_padding = self._processed_data.cell_size_x / 2
        y_padding = self._processed_data.cell_size_y / 2

        # Set view of axes
        self.ax.set_xlim([
            self._processed_data.rounded_x_min - x_padding,
            self._processed_data.rounded_x_max + x_padding,
        ])
        self.ax.set_ylim([
            self._processed_data.rounded_y_min - y_padding,
            self._processed_data.rounded_y_max + y_padding,
        ])

        # Set background checkerboard:
        self.__set_background(
            self._processed_data.rounded_x_min,
            self._processed_data.rounded_y_min,
            self._processed_data.cell_size_x,
            self._processed_data.cell_size_y,
            self._processed_data.rounded_x_max,
            self._processed_data.rounded_y_max,
        )

    def __process(self, trajectory):
        """Get relevant data (and do merging of repeated states if desired)"""

        if not trajectory.process_data():
            return  # early return for cases where trajectory data already processed

        # used for deciding node sizes
        self._processed_data.max_duration = max(
            max(trajectory.processed_data.nodes),
            self._processed_data.max_duration,
        )

        # Get min and max values
        x_min, x_max = util.calculate_min_max(trajectory.processed_data.x)
        y_min, y_max = util.calculate_min_max(trajectory.processed_data.y)

        # todo :: ?? unsure about what some of the following defaulting wants to do.
        if self.style.x_minmax_given[0]:
            if self._processed_data.x_min is None:
                self._processed_data.x_min = self.style.x_min
        elif self._processed_data.x_min is None or x_min < self._processed_data.x_min:
            self._processed_data.x_min = x_min

        if self.style.x_minmax_given[1]:
            if self._processed_data.x_max is None:
                self._processed_data.x_max = self.style.x_max
        elif self._processed_data.x_max is None or x_max > self._processed_data.x_max:
            self._processed_data.x_max = x_max

        if self.style.y_minmax_given[0]:
            if self._processed_data.y_min is None:
                self._processed_data.y_min = self.style.y_min
        elif self._processed_data.y_min is None or y_min < self._processed_data.y_min:
            self._processed_data.y_min = y_min

        if self.style.y_minmax_given[1]:
            if self._processed_data.y_max is None:
                self._processed_data.y_max = self.style.y_max
        elif self._processed_data.y_max is None or y_max > self._processed_data.y_max:
            self._processed_data.y_max = y_max

    def set_style(self, grid_style: GridStyle):
        self._processed_data = GridCumulativeData()  # clear
        self.style = grid_style

    def add_trajectory_data(self, *trajectories: Trajectory):
        if not trajectories:
            # todo :: do we really care...?
            raise ValueError("A list of trajectories must be supplied to Grid")

        # check all trajectories before we add things:
        # todo :: uhhhh need to see if this is really the right logic
        # todo :: or way to structure this
        meta_names = set(
            self.trajectory_list[0].meta.keys()
            if self.trajectory_list
            else trajectories[0].meta.keys()
        )
        for trajectory in trajectories:
            assert isinstance(trajectory, Trajectory), (
                "All trajectory data supplied in trajectory list"
                " must be instances of the trajectory class"
            )
            # todo :: possibly go for a more type-agnostic approach here? (see above)
            assert len(meta_names) == len(trajectory.meta), (
                f"trajectory ID {trajectory.id}:"
                " metadata fields don't match with others in list"
            )
            for i in trajectory.meta:
                assert i in meta_names, (
                    f"metadata field name {i} in trajectory ID {trajectory.id}"
                    " does not exist in first trajectory"
                )
            # todo :: do we also want to check id?

        # todo :: (??)
        # they look good, so add them:
        for trajectory in trajectories:
            self.trajectory_list[trajectory.id] = trajectory
        self._processed_data = GridCumulativeData()  # clear

    def draw(self, save_as: Optional[str] = None):
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
        if save_as is not None:
            self.ax.savefig(save_as)
        else:
            plt.show()
        self._processed_data.valid = True

    def __calculate_dispersion(self):
        n = (
            int(
                (self._processed_data.x_max - self._processed_data.x_min)
                / self._processed_data.cell_size_x
            ) + 1
        ) * (
            int(
                (self._processed_data.y_max - self._processed_data.y_min)
                / self._processed_data.cell_size_y
            ) + 1
        )
        cell_durations = {}
        total_duration = 0
        for traj in self.trajectory_list:
            for duration, x, y in zip(
                traj.processed_data.nodes,
                traj.processed_data.x,
                traj.processed_data.y,
            ):
                # todo :: defaultdict
                if y in cell_durations:
                    cell_durations[y][x] = cell_durations[y].get(x, 0) + duration
                else:
                    cell_durations[y] = {x: duration}
                total_duration += duration
        # todo :: why fractions? :)
        sum_d_D = sum(
            pow(fractions.Fraction(d, total_duration), 2)
            for x_and_d in cell_durations.values()
            for d in x_and_d.values()
        )
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

        durations = [traj.get_duration() for traj in self.trajectory_list]
        event_numbers = [len(traj.data_x) for traj in self.trajectory_list]
        visit_numbers = [traj.get_num_visits() for traj in self.trajectory_list]
        cell_ranges = [traj.get_cell_range() for traj in self.trajectory_list]

        # todo :: hmmmmmmMMMMM
        return GridMeasures(
            trajectory_ids=[traj.id for traj in self.trajectory_list],
            mean_duration=mean(durations),
            mean_number_of_events=mean(event_numbers),
            mean_number_of_visits=mean(visit_numbers),
            mean_cell_range=mean(cell_ranges),
            overall_cell_range=sum(
                len(x_and_count)
                for x_and_count in self._processed_data.bin_counts.items()
            ),
            mean_duration_per_event=mean(
                map(lambda x, y: x / y, durations, event_numbers)
            ),
            mean_duration_per_visit=mean(
                map(lambda x, y: x / y, durations, visit_numbers)
            ),
            mean_duration_per_cell=mean(
                map(lambda x, y: x / y, durations, cell_ranges)
            ),
            dispersion=float(self.__calculate_dispersion()),
        )
