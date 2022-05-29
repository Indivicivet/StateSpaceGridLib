import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import numpy as np
from .Trajectory import Trajectory, Trajectorystyle, check_trajectory_list
from .States import *


class Gridstyle:
    def __init__(self, title="", labelfontsize=14, tickfontsize=14, titlefontsize=14, tick_increment_x = None, tick_increment_y = None,
                 x_label = None, y_label = None, x_order = None, y_order = None, x_min = None, x_max = None, y_min = None, y_max = None,
                 rotate_xlabels=False):
        self.labelfontsize = labelfontsize
        self.tickfontsize = tickfontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.tick_increment_x = tick_increment_x
        self.tick_increment_y = tick_increment_y
        self.x_label = x_label
        self.y_label = y_label
        self.x_order = x_order
        self.y_order = y_order
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.x_minmax_given = (x_min is not None,x_max is not None)
        self.y_minmax_given = (y_min is not None,y_max is not None)
        self.rotate_xlabels=rotate_xlabels


class Grid:
    def __init__(self, trajectories, style=Gridstyle()):
        self.trajectory_list = [i for i in trajectories]
        self.graph = nx.Graph()
        self.ax = plt.gca()
        self.style = style
        check_trajectory_list(self.trajectory_list)
    
    def __get_data(self, trajectory):
        loop_nodes = set()
        # expect last field to be NaN due to 1 extra time field
        x_data, y_data, time_data = [], [], []
        if trajectory.style.merge_repeated_states:
            loop_nodes, x_data, y_data, time_data = trajectory.merge_equal_adjacent_states()
        else:
            time_data = trajectory.data_t
            x_data = trajectory.data_x
            y_data = trajectory.data_y
        return loop_nodes, x_data, y_data, time_data

    def __set_background(self, x_min, y_min, x_scale, y_scale, x_max, y_max):
        background_colours = ListedColormap([np.array([220 / 256, 220 / 256, 220 / 256, 1]), np.array([1, 1, 1, 1])])
        background = [[((i + j) % 2) for i in range(int(x_min/x_scale), int(x_max/x_scale)+1)] for j in range(int(y_min/y_scale), int(y_max/y_scale)+1)]
        self.ax.imshow(background, extent=[int(x_min)-0.5*x_scale, int(x_max)+0.5*x_scale, int(y_min)-0.5*y_scale, int(y_max)+0.5*y_scale], cmap=background_colours, interpolation='none')
    
    def __draw_graph(self, x_data, y_data, time_data, loop_nodes, drawstyle):
        # Create a dictionary to define positions for node numbers
        pos = {i: (x_data[i], y_data[i]) for i in range(len(x_data))}

        # List of tuples to define edges between nodes
        edges = [(i, i + 1) for i in range(len(x_data) - 1)]
        for loop_node in loop_nodes:
            edges.append((loop_node, loop_node))

        # Calculate node size based on time data
        node_sizes = [(time_data[i + 1] - time_data[i]) for i in range(len(time_data) - 1)]
        node_size_scale_factor = 1000 / max(node_sizes)
        for i in range(len(node_sizes)):
            node_sizes[i] = node_size_scale_factor * node_sizes[i]

        # Add nodes and edges to graph
        self.graph.add_nodes_from(pos.keys())
        self.graph.add_edges_from(edges)

        # Draw graphs
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color='indigo')
        nx.draw_networkx_edges(self.graph, pos, node_size=node_sizes, nodelist=[i for i in range(len(x_data))],
                                       edgelist=edges, arrows=True, arrowstyle=drawstyle.arrow_style, node_shape='.',
                                       arrowsize=10, width=2, connectionstyle=drawstyle.connection_style, )

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
                                                                                 range(self.style.x_min,
                                                                                       self.style.x_max + 1, self.style.tick_increment_x)]
        if self.style.y_order:
            tick_label_y = self.style.y_order
        else:
            tick_label_y = drawstyle.ordering["y"] if "y" in drawstyle.ordering else [str(i) for i in
                                                                                                 range(self.style.y_min,
                                                                                                       self.style.y_max + 1, self.style.tick_increment_y)]
        # Set ticks for states
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.tick_params(axis='x', labelsize=self.style.tickfontsize, rotation=90 if self.style.rotate_xlabels else 0)
        self.ax.tick_params(axis='y', labelsize=self.style.tickfontsize)
        self.ax.xaxis.set_major_locator(ticker.FixedLocator([i for i in range(self.style.x_min, self.style.x_max + 1)]))
        self.ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_label_x))
        self.ax.yaxis.set_major_locator(ticker.FixedLocator([i for i in range(self.style.y_min, self.style.y_max + 1)]))
        self.ax.yaxis.set_major_formatter(ticker.FixedFormatter(tick_label_y))

        # Set axis labels
        self.ax.set_xlabel(self.style.x_label, fontsize=self.style.labelfontsize)
        self.ax.set_ylabel(self.style.y_label, fontsize=self.style.labelfontsize)

        if self.style.title:
            self.ax.set_title(self.style.title, fontsize=self.style.titlefontsize)

    def __draw_background_and_view(self):
        # Make an estimate for scale size of checkerboard grid sizing
        x_scale = calculate_scale(self.style.x_min, self.style.x_max) if self.style.tick_increment_x is None else self.style.tick_increment_x
        y_scale = calculate_scale(self.style.y_min, self.style.y_max) if self.style.tick_increment_y is None else self.style.tick_increment_y

        # round x min and x_max for drawing axes
        # TODO
        # we should probably put this stuff in a seperate "__calculated_data" struct
        self.style.x_min -= (self.style.x_min % x_scale)
        self.style.x_max += (x_scale - ((self.style.x_max % x_scale) if self.style.x_max % x_scale else x_scale))
        self.style.y_min -= (self.style.y_min % y_scale)
        self.style.y_max += (y_scale - ((self.style.y_max % y_scale) if self.style.y_max % y_scale else y_scale))

        x_padding = x_scale / 2
        y_padding = y_scale / 2

        # Set view of axes
        self.ax.set_xlim([self.style.x_min - x_padding, self.style.x_max + x_padding])
        self.ax.set_ylim([self.style.y_min - y_padding, self.style.y_max + y_padding])

        # Set background checkerboard:
        self.__set_background(self.style.x_min, self.style.y_min, x_scale, y_scale, self.style.x_max, self.style.y_max)

    def __add_plot(self, trajectory):
        # Get relevant data (and do merging of repeated states if desired)
        loop_nodes, x_data, y_data, time_data = self.__get_data(trajectory)

        # Get min and max values
        x_min, x_max = calculate_min_max(x_data)
        y_min, y_max = calculate_min_max(y_data)
        if self.style.x_minmax_given[0]:
            x_min = self.style.x_min
        else:
            # if x_min not given, we should keep track of this
            x_min = x_min if self.style.x_min is None else min(x_min,self.style.x_min)
            self.style.x_min = x_min
        if self.style.x_minmax_given[1]:
            x_max = self.style.x_max
        else:
            x_max = x_max if self.style.x_max is None else max(x_max,self.style.x_max)
            self.style.x_max = x_max
        if self.style.y_minmax_given[0]:
            y_min = self.style.y_min
        else:
            # if y_min not given, we should keep track of this
            y_min = y_min if self.style.y_min is None else min(y_min,self.style.y_min)
            self.style.y_min = y_min
        if self.style.y_minmax_given[1]:
            y_max = self.style.y_max
        else:
            y_max = y_max if self.style.y_max is None else max(y_max,self.style.y_max)
            self.style.y_max = y_max

        # Make an estimate for scale size of checkerboard grid sizing
        x_scale = calculate_scale(x_min, x_max) if not self.style.tick_increment_x else self.style.tick_increment_x
        y_scale = calculate_scale(y_min, y_max) if not self.style.tick_increment_y else self.style.tick_increment_y

        if not self.style.tick_increment_x:
            self.style.tick_increment_x = x_scale
        if not self.style.tick_increment_y:
            self.style.tick_increment_y = y_scale

        # If same state is repeated, offset states so they don't sit on top of one another:
        offset_within_bin(x_data, x_scale, y_data, y_scale)

        self.__draw_graph(x_data, y_data, time_data, loop_nodes, trajectory.style)

    def set_style(self, gridstyle: Gridstyle):
        self.style = gridstyle

    def get_style(self):
        return self.style

    def add_trajectory_data(self, *trajectories: Trajectory):
        for trajectory in trajectories:
            self.trajectory_list[trajectory.meta["ID"]] = trajectory
        if trajectories:
            check_trajectory_list(self.trajectory_list)

    def draw(self):
        for trajectory in self.trajectory_list:
            self.__add_plot(trajectory)
        self.__draw_background_and_view()
        self.__draw_ticks(self.trajectory_list[0].style)
        self.ax.set_aspect('auto')
        plt.tight_layout()
        plt.show()



