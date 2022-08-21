from pathlib import Path

import pandas as pd

import state_space_grid as ssg


DATA_PATH = Path(__file__).resolve().parent / "resources" / "ExampleData1.txt"


# todo :: sensible name
def test1():
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


# todo :: sensible name
def test2():
    data1 = pd.read_csv(DATA_PATH.open())
    traj1 = ssg.Trajectory(
        data1["variable 1"].tolist(),
        data1["variable 2"].tolist(),
        data1["Onset"].tolist(),
    )
    # traj1.add_y_ordering(["Low", "Medium", "High"])
    grid = ssg.Grid(
        [traj1],
        style=ssg.GridStyle(
            title="test 2",
            x_label="variable 1",
            y_label ="variable 2",
            title_font_size=30,
            x_max=4,
            x_min=1,
            y_max=4,
            y_min=1
        ),
    )
    # grid.draw(
    print(grid.get_measures())


if __name__ == '__main__':
    test2()
