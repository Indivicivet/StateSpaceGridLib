from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Sequence
from numbers import Number
from statistics import mean

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import numpy as np

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
    x_order: List[Union[str, Number]] = field(default_factory=list)
    y_order: List[Union[str, Number]] = field(default_factory=list)
    rotate_x_labels: bool = False


@dataclass
class GridMeasures:
    # todo :: all of these fields probably want to be methods
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


@dataclass
class Grid:
    trajectory_list: List[Trajectory]
    style: GridStyle = field(default_factory=GridStyle)

    def shared_all_trajectory_process(self):
        # todo :: sensible name :)
        max_duration = 0
        x_min = self.trajectory_list[0].data_x[0]
        y_min = self.trajectory_list[0].data_y[0]
        x_max = x_min
        y_max = y_min
        loops_list = []

        for trajectory in self.trajectory_list:
            x_data, y_data, t_data, loops = trajectory.get_states(self.style.x_order, self.style.y_order)

            # Get min and max values
            # todo :: this is probably silly...
            temp_x_min, temp_x_max = util.calculate_min_max(x_data)
            temp_y_min, temp_y_max = util.calculate_min_max(y_data)

            if self.style.x_min is not None:
                x_min = self.style.x_min
            if self.style.x_max is not None:
                x_max = self.style.x_max

            if self.style.y_min is not None:
                y_min = self.style.y_min
            if self.style.y_max is not None:
                y_max = self.style.y_max

            max_duration = max(*(t2 - t1 for t1, t2 in zip(t_data, t_data[1:])), max_duration)
            x_min = min(x_min, int(temp_x_min))
            y_min = min(y_min, int(temp_y_min))
            x_max = max(x_max, int(temp_x_max))
            y_max = max(y_max, int(temp_y_max))
            # todo :: is there a necessary loops reset I've accidentally removed?
            loops_list.append(loops)
        return max_duration, x_min, y_min, x_max, y_max, loops_list

    def get_rounded_parameters(self, x_min, x_max, y_min, y_max):
        """returns:
        (
            cell width,
            cell height,
            rounded x min,
            rounded y min,
            rounded x max,
            rounded y max,
        )
        """
        cell_size_x = (
            util.calculate_scale(x_max - x_min)
            if self.style.tick_increment_x is None
            else self.style.tick_increment_x
        )
        cell_size_y = (
            util.calculate_scale(y_max - y_min)
            if self.style.tick_increment_y is None
            else self.style.tick_increment_y
        )
        return (
            cell_size_x,
            cell_size_y,
            x_min - (x_min % cell_size_x),
            y_min - (y_min % cell_size_y),
            # todo :: what is this surely there's builtins
            x_max + cell_size_x - (
                x_max % cell_size_x
                if x_max % cell_size_x
                else cell_size_x
            ),
            y_max + cell_size_y - (
                y_max % cell_size_y
                if y_max % cell_size_y
                else cell_size_y
            ),
        )

    def draw(self, save_as: Optional[str] = None):
        graph = nx.Graph()
        ax = plt.gca()

        max_duration, x_min, y_min, x_max, y_max, loops_list = self.shared_all_trajectory_process()

        cell_size_x, cell_size_y, rounded_x_min, rounded_y_min, rounded_x_max, rounded_y_max \
            = self.get_rounded_parameters(x_min, x_max, y_min, y_max)

        # draw background
        # todo :: whole bunch of stuff that is a bit messy here
        # Make an estimate for scale size of checkerboard grid sizing

        x_padding = cell_size_x / 2
        y_padding = cell_size_y / 2

        # Set view of axes
        ax.set_xlim([
            rounded_x_min - x_padding,
            rounded_x_max + x_padding,
        ])
        ax.set_ylim([
            rounded_y_min - y_padding,
            rounded_y_max + y_padding,
        ])

        # Set background checkerboard:
        jj, ii = np.mgrid[
                 int(rounded_y_min / cell_size_y):int(rounded_y_max / cell_size_y) + 1,
                 int(rounded_x_min / cell_size_x):int(rounded_x_max / cell_size_x) + 1,
                 ]
        ax.imshow(
            # checkerboard
            (jj + ii) % 2,
            extent=[
                int(rounded_x_min) - 0.5 * cell_size_x,
                int(rounded_x_max) + 0.5 * cell_size_x,
                int(rounded_y_min) - 0.5 * cell_size_y,
                int(rounded_y_max) + 0.5 * cell_size_y,
            ],
            cmap=ListedColormap([
                [220 / 256] * 3,
                [1] * 3,
            ]),
            interpolation='none',
        )

        # now we draw trajectories
        offset = offset_trajectories(self.trajectory_list, self.style, cell_size_x, cell_size_y)

        for offset_trajectory, loops in zip(offset, loops_list):
            draw_graph(offset_trajectory, loops, self.style, graph, max_duration)

        # all of this needs to go in a separate function, called with show()
        # we need to store which axis is which column - kick up a fuss if future plots don't match this
        # we also need to store an idea of minimum scale - this is going to fuck us if scales don't match
        # - maybe just tell people if they have scale adjustment problems to specify their own

        # Get tick labels - either numeric or categories
        # todo :: this bit is a bit of a mess
        rounded_x_points = range(
            int(rounded_x_min),
            int(rounded_x_max + 1),
            int(cell_size_x),
        )
        rounded_y_points = range(
            int(rounded_y_min),
            int(rounded_y_max + 1),
            int(cell_size_y),
        )
        tick_label_x = (
                self.style.x_order
                or [str(i) for i in rounded_x_points]
        )
        tick_label_y = (
                self.style.y_order
                or [str(i) for i in rounded_y_points]
        )
        # Set ticks for states
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params(
            axis='x',
            labelsize=self.style.tick_font_size,
            rotation=90 * self.style.rotate_x_labels,
        )
        ax.tick_params(
            axis='y',
            labelsize=self.style.tick_font_size,
        )
        ax.xaxis.set_major_locator(ticker.FixedLocator(rounded_x_points))
        ax.yaxis.set_major_locator(ticker.FixedLocator(rounded_y_points))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_label_x))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(tick_label_y))

        # Set axis labels
        ax.set_xlabel(self.style.x_label, fontsize=self.style.label_font_size)
        ax.set_ylabel(self.style.y_label, fontsize=self.style.label_font_size)

        if self.style.title:
            ax.set_title(self.style.title, fontsize=self.style.title_font_size)

        ax.set_aspect('auto')
        plt.tight_layout()

        if save_as is not None:
            ax.savefig(save_as)
        else:
            plt.show()

    def get_measures(self):
        max_duration, x_min, y_min, x_max, y_max, _ = self.shared_all_trajectory_process()

        cell_size_x, cell_size_y, rounded_x_min, rounded_y_min, rounded_x_max, rounded_y_max \
            = self.get_rounded_parameters(x_min, x_max, y_min, y_max)

        trajectory_durations, event_numbers, visit_numbers, cell_ranges = zip(*[
            (
                traj.data_t[-1] - traj.data_t[0],
                len(traj.data_x),
                traj.get_num_visits(),
                traj.get_cell_range()
            )
            for traj in self.trajectory_list
        ])

        def maybe_reorder(data, ordering: Optional[list] = None):
            # todo :: this is expected behaviour ordering is Falsey (but not none) - investigate
            # (because it seems like it should only do nothing if ordering=None...)
            if not ordering:
                return data
            return [ordering.index(val) for val in data]

        bin_counts = Counter()
        for trajectory in self.trajectory_list:
            bin_counts += Counter(
                zip(
                    maybe_reorder(trajectory.data_x, self.style.x_order),
                    maybe_reorder(trajectory.data_y, self.style.y_order),
                )
            )

        # todo :: hmmmmmmMMMMM
        return GridMeasures(
            trajectory_ids=[trajectory.id for trajectory in self.trajectory_list],
            mean_duration=mean(trajectory_durations),
            mean_number_of_events=mean(event_numbers),
            mean_number_of_visits=mean(visit_numbers),
            mean_cell_range=mean(cell_ranges),
            overall_cell_range=sum(1 for x_and_count in bin_counts.values()),
            # todo :: these should likely be the responsibility of GridMeasures
            mean_duration_per_event=mean(
                map(lambda x, y: x / y, trajectory_durations, event_numbers)
            ),
            mean_duration_per_visit=mean(
                map(lambda x, y: x / y, trajectory_durations, visit_numbers)
            ),
            mean_duration_per_cell=mean(
                map(lambda x, y: x / y, trajectory_durations, cell_ranges)
            ),
            dispersion=mean(
                trajectory.calculate_dispersion(
                    # todo :: what is this nonsense :)
                    (int((x_max - x_min) / cell_size_x) + 1) * (int((y_max - y_min) / cell_size_y) + 1),
                )
                for trajectory in self.trajectory_list
            ),
        )


def offset_trajectories(
    trajectories: Sequence[Trajectory],
    grid_style: GridStyle,
    cell_size_x: float,  # todo :: I actually don't know what these are
    cell_size_y: float,  # todo :: I actually don't know what these are
) -> list[Trajectory]:
    # todo :: later -- need to think about what this actually does
    # get total bin counts
    new_trajectories = []
    # todo :: are we independently getting bin counts in many places...
    bin_counts = Counter(
        (x, y)
        for trajectory in trajectories
        for (x, y) in zip(*trajectory.get_states(grid_style.x_order, grid_style.y_order)[:2])
    )
    current_bin_counter = Counter()
    for trajectory in trajectories:
        # If same state is repeated, offset states
        # so they don't sit on top of one another:
        x_data, y_data, t_data, _ = trajectory.get_states(grid_style.x_order, grid_style.y_order)
        new_trajectories.append(
            Trajectory(
                *util.offset_within_bin(
                    x_data,
                    cell_size_x,
                    y_data,
                    cell_size_y,
                    bin_counts,
                    current_bin_counter,
                ),
                t_data,
                trajectory.meta,
                trajectory.style,
            )
        )
    return new_trajectories


def draw_graph(trajectory, loops, grid_style, graph, max_duration):
    x_data, y_data, t_data, _ = trajectory.get_states(grid_style.x_order, grid_style.y_order)
    node_number_positions = dict(enumerate(zip(x_data, y_data)))

    # List of tuples to define edges between nodes
    edges = (
            [(i, i + 1) for i in range(len(x_data) - 1)]
            + [(loop_node, loop_node) for loop_node in loops]
    )

    durations = list(t2 - t1 for t1, t2 in zip(t_data, t_data[1:]))

    node_sizes = list(  # todo :: maybe redundant list()
        (1000 / max_duration)
        * np.array(durations)
    )

    # Add nodes and edges to graph
    graph.add_nodes_from(node_number_positions.keys())
    graph.add_edges_from(edges)

    # Draw graphs
    nx.draw_networkx_nodes(graph, node_number_positions, node_size=node_sizes, node_color='indigo')
    nx.draw_networkx_edges(
        graph,
        node_number_positions,
        node_size=node_sizes,
        nodelist=list(range(len(x_data))),
        edgelist=edges,
        arrows=True,
        arrowstyle=trajectory.style.arrow_style,
        node_shape='.',
        arrowsize=10,
        width=2,
        connectionstyle=trajectory.style.connection_style,
    )
