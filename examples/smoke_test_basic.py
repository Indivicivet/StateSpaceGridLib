from pathlib import Path

import pandas as pd

import state_space_grid as ssg

DATA_PATH = Path(__file__).resolve().parent / "resources" / "ExampleData1.txt"


# todo :: sensible name
def test_1():
    data1 = pd.read_csv(DATA_PATH.open())
    traj1 = ssg.Trajectory(
        data1["variable 1"].dropna().tolist(),
        data1["variable 2"].dropna().tolist(),
        data1["Onset"].dropna().tolist(),
    )
    grid = ssg.Grid(
        [traj1],
        style=ssg.GridStyle(x_label="variable 1", y_label="variable 2"),
    )
    grid.draw()


if __name__ == '__main__':
    test_2()
