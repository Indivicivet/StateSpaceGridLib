from pathlib import Path

import pandas as pd

from state_space_grid import grid, trajectory


def plot_example_data(data_path):
    data1 = pd.read_csv(data_path.open())
    traj1 = trajectory.Trajectory(
        data1["variable 1"].dropna().tolist(),
        data1["variable 2"].dropna().tolist(),
        data1["Onset"].dropna().tolist(),
    )
    my_grid = grid.Grid(
        [traj1],
    )
    my_grid.draw(style=grid.GridStyle(x_label="variable 1", y_label="variable 2"))


if __name__ == '__main__':
    DATA_PATH = Path(__file__).resolve().parent / "resources" / "ExampleData1.txt"
    plot_example_data(DATA_PATH)
