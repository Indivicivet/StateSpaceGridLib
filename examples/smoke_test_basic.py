from pathlib import Path

import pandas as pd

from state_space_grid import grid, trajectory

DATA_PATH = Path(__file__).resolve().parent / "resources" / "ExampleData1.txt"


# todo :: sensible name
def test_1():
    data1 = pd.read_csv(DATA_PATH.open())
    traj1 = trajectory.Trajectory(
        data1["variable 1"].dropna().tolist(),
        data1["variable 2"].dropna().tolist(),
        data1["Onset"].dropna().tolist(),
    )
    my_grid = grid.Grid(
        [traj1],
        style=grid.GridStyle(x_label="variable 1", y_label="variable 2"),
    )
    my_grid.draw()


if __name__ == '__main__':
    test_2()
