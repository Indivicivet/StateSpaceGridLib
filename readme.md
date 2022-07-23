# StateSpaceGridLib

## Introduction

StateSpaceGrid is a python library loosely based around replicating the
functionality of GridWare, a Java program for creating and analysing
state space grids for visualising dyadic mutuality in psychology.

The main features of StateSpaceGrid are Trajectory objects, Grid objects and
their associated styles.

Trajectory objects hold the input data as a set of lists as well as information 
about the visualisation of a single graph (line style, colour, etc.).

Grid objects hold trajectory objects and are the interface through which you
can display the graphs.

A brief test is included to display the code in action.

To install as a library with pip, use

pip install git+https://github.com/maggym00/StateSpaceGridLib

## Library API Reference

#### Trajectory

```python
Trajectory(data_x, data_y, data_t, meta=None, style=Trajectorystyle())
```
Data in StateSpaceGridLib is organised via Trajectories. These objects take in data for the x axis of the grid, data for the y axis of the grid, and time data for the transitions between states (ie. the time data specifies the time at which each state starts, as well as the end time of the last state).
* `data_x`

   A list of state values for the x axis. The length of this list is expected to be the same as for the y data.

* `data_y`

   A list of state values for the y axis. The length of this list is expected to be the same as for the x data.
* `data_t`

   Time data: a list of length 1 longer than the x or y data, specifying the start time of each event in the data_x/data_y lists as well as the end point of the final event.

* `meta`

   A `dictionary` containing extra information about data going into the `Trajectory` for annotating the trajectory and ease of lookup. At the very least this should contain an `"ID"` field - if one is not given, then one will be provided in the form of the count of `Trajectory` objects at point of creation. 

* `style`

   A `TrajectoryStyle` object containing settings for visualisation of `Trajectories` as well as any information about ordering of non-numeric measurement scales (or unconventionally ordered numerical ones).

   A `TrajectoryStyle` object intended for the `Trajectory` object.

```python
Trajectory.add_x_ordering(ordering)
```
Set ordering for x measurement scale.
* `ordering`

   A list going from lowest possible measurement value to highest.
   
   Eg. `["low", "medium", "high"]`

```python
Trajectory.add_y_ordering(ordering)
```
Set ordering for y measurement scale.
* `ordering`

   A list going from lowest possible measurement value to highest.
   
   Eg. `["low", "medium", "high"]`

```python
Trajectory.add_global_ordering(ordering)
```
Set ordering for both y and x measurement scale.
* `ordering`

   A list going from lowest possible measurement value to highest.
   
   Eg. `["low", "medium", "high"]`

```python
Trajectory.get_duration()
```
Return duration of trajectory (ie. `data_t[-1] - data_t[0]`)

```python
Trajectory.get_num_visits()
```
Return number of "visits", defined as the number of state transitions plus 1 (the initial starting state) minus the number of transitions to the same state as where the transition is from (ie. `(x1, y1) -> (x2, y2)` where  .

#### TrajectoryStyle
```python
TrajectoryStyle(connectionstyle="arc3,rad=0.0", arrowstyle='-|>', ordering = dict(), merge_repeated_states=True)
```
Object to customise visualisation and the measurement value ordering for individual trajectories in grids.
* `connectionstyle`

   Controls the style of the lines connecting states on the grid. See [matplotlib.patches.ConnectionStyle](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.ConnectionStyle.html#matplotlib.patches.ConnectionStyle) and [matplotlib.patches.FancyArrowPatch](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch) for more information.
* `arrowstyle`

   Controls the style of the arrows showing direction of transition on lines between states. See [matplotlib.patches.ArrowStyle](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.ArrowStyle.html#matplotlib.patches.ArrowStyle) for more information.
* `ordering`

   Controls the ordering of x and y data measurement scales. Data is expected in the form 
   
   `{"x":(x_1, x_2,...), "y": (y1, y2,...)}`
* `merge_repeated_states`

   Controls whether or not adjacent events with repeated states should be merged and displayed on a grid as a single event with a loop coming out and returning to the event node to signify that a merge has occurred.
   
```python
TrajectoryStyle.add_ordering(axis, ordering)
```
Sets the ordering for the specified axis.
* `axis`

   axis for which the ordering applies - expected values are "x" or "y".
* `ordering`

   A list going from lowest possible measurement value to highest.
   
   Eg. `["low", "medium", "high"]`

#### Grid
```python
Grid(trajectories, style=Gridstyle())
```
Data now formatted as `Trajectory` objects can be collated within a `Grid` object for visualisation and the calculation of grid measures.
* `trajectories`
   
   A list of `Trajectory` objects containing the input data.

* `style`

   A `GridStyle` object containing information about the formatting of the grid visualisation including axis labels, and titles, as well as data about grid-wide data ordering, minimum and maximum possible values, and scale increments (ie. to handle cases such as scales being from 0-100 going up in multiples of 10)
```python
Grid.setStyle(style)
```
Set `GridStyle` object stored in `Grid.style`
* `style`

   A GridStyle object intended for the `Grid` object.
```python
Grid.getStyle()
```
Get `GridStyle` object stored in `Grid.style`.
```python
Grid.add_trajectory_data(trajectory1, trajectory2,...)
```
Add `Trajectory` objects to the grid. 
* `trajectory<n>`

   `Trajectory` object to be added
```python
Grid.draw(save_as="")
```
Draw visualisation of state space grid.
* `save_as`

   If provided, this is the name of the png file that the image of the grid is saved as.  If left blank, the grid will be displayed in a new window using whichever gui library matplotlib has access to.
```python
Grid.get_measures()
```
Calculate cumulative measures from all trajectories provided to the grid and return as a `GridMeasures` object.
#### GridStyle
```python
GridStyle(title="", label_font_size=14, tick_font_size=14, title_font_size=14, tick_increment_x=None, tick_increment_y=None, x_label=None, y_label=None, x_order=None, y_order=None, x_min=None, x_max=None, y_min=None, y_max=None, rotate_xlabels=False)
```
Object containing visualisation customisation and global controls for trajectory data scales.
* `title`

   Grid title
* `label_font_size`

   Axis label font size
* `tick_font_size`

   Axis tick font size
* `title_font_size`

   Title font size
* `tick_increment_x`

   X scale increment
* `tick_increment_y`

   Y scale increment
* `x_label`

   X axis label
* `y_label`

   Y axis label
* `x_order`

   Grid-wide x data scale ordering.
   
   A list going from lowest possible measurement value to highest.
   
   Eg. `["low", "medium", "high"]`
   
   (overrides any information contained within `TrajectoryStyle` ordering)
* `y_order`

   Grid-wide y data scale ordering.
   
   A list going from lowest possible measurement value to highest.
   
   Eg. `["low", "medium", "high"]`
   
   (overrides any information contained within `TrajectoryStyle` ordering)
* `x_min` 
   
   Minimum possible value in x data scale.
* `x_max` 
   
   Maximum possible value in x data scale.
* `y_min` 
   
   Minimum possible value in y data scale.
* `y_max` 
   
   Maximum possible value in y data scale.
* `rotate_xlabels`

   Rotate x axis tick labels by 90°.
#### GridMeasures
```python
GridMeasures()    
```
Data container for all measure data. Individual values can be read off, or the entire thing may be printed as is - the class contains a `__str()__` function for ease of quickly printing the data.
```python
GridMeasures.trajectory_ids
```
IDs of all trajectories in grid.
```python
GridMeasures.mean_duration
```
Mean total duration of the trajectories
```python
GridMeasures.mean_number_of_events
```
Mean number of events in the trajectories
```python
GridMeasures.mean_number_of_visits
```
Mean number of visits in the trajectories.

A visit is defined as being an entrance and then an exit of a cell.
```python
GridMeasures.mean_cell_range
```
Mean number of different cells reached by the trajectories.
```python
GridMeasures.overall_cell_range
```
Total number of cells visited cumulatively by the trajectories. 
```python
GridMeasures.mean_duration_per_event
```
Mean duration per event

Defined as mean of trajectory duration divided by number of trajectory events.
```python
GridMeasures.mean_duration_per_visit
```
Mean duration per visit

Defined as mean of trajectory duration divided by number of trajectory visits.
```python
GridMeasures.mean_duration_per_cell
```
Mean duration spent in each cell

Defined as mean of trajectory duration divided by trajectory cell range.
```python
GridMeasures.dispersion
```
Dispersion calculated across all trajectories.
