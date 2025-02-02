from pathlib import Path

import pandas as pd

from state_space_grid import grid, trajectory


RESOURCES = Path(__file__).resolve().parent.parent / "examples" / "resources"


# todo :: the following used to exist, is there any sensible way to test it? :)
"""
style = grid.GridStyle(
    title="test 2",
    x_label="variable 1",
    y_label="variable 2",
    title_font_size=30,
),
"""


def test_ExampleData1_measures():
    data1 = pd.read_csv((RESOURCES / "ExampleData1.txt").open())
    traj1 = trajectory.Trajectory(
        data1["variable 1"].dropna().tolist(),
        data1["variable 2"].dropna().tolist(),
        data1["Onset"].dropna().tolist(),
    )
    # traj1.add_y_ordering(["Low", "Medium", "High"])
    my_grid = grid.Grid(
        [traj1],
        quantization=grid.GridQuantization(
            x_max=4,
            x_min=1,
            y_max=4,
            y_min=1,
        )
    )
    measures = my_grid.get_measures()
    assert measures.mean_duration == 14.8
    assert measures.mean_number_of_events == 8
    assert measures.mean_number_of_visits == 7
    assert measures.mean_cell_range == 6
    assert measures.overall_cell_range == 6
    assert measures.mean_duration_per_event == 1.85
    assert abs(measures.mean_duration_per_visit - 2.1142857) < 1e-3
    assert abs(measures.mean_duration_per_cell - 2.46666) < 1e-3
    assert abs(measures.dispersion - 0.933333) < 1e-3
    assert measures.mean_missing_events == 0
    assert measures.mean_missing_duration == 0


def test_ExampleData_measures_from_legacy_trj():
    traj1 = trajectory.Trajectory.from_legacy_trj(RESOURCES / "ExampleData.trj")
    # traj1.add_y_ordering(["Low", "Medium", "High"])
    my_grid = grid.Grid(
        [traj1],
        quantization=grid.GridQuantization(
            x_max=4,
            x_min=1,
            y_max=4,
            y_min=1,
        )
    )
    measures = my_grid.get_measures()
    assert measures.mean_duration == 14.0
    assert measures.mean_number_of_events == 9
    assert measures.mean_number_of_visits == 9
    assert measures.mean_cell_range == 6
    assert measures.overall_cell_range == 6
    assert abs(measures.mean_duration_per_event - 1.5555555) < 1e-3
    assert abs(measures.mean_duration_per_visit - 1.5555555) < 1e-3
    assert abs(measures.mean_duration_per_cell - 2.3333333) < 1e-3
    assert abs(measures.dispersion - 0.9481481) < 1e-3
    assert measures.mean_missing_events == 0
    assert measures.mean_missing_duration == 0
