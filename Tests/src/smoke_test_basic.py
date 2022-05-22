import StateSpaceGrid as ssg
import pandas as pd
import os

def test1():
    file_path = os.path.dirname(__file__)
    data1 = pd.read_csv(open((file_path+"\\" if file_path else "") + r"..\resources\ExampleData1.txt"))
    traj1 = ssg.Trajectory(data1["variable 1"].tolist(),data1["variable 2"].tolist(), data1["Onset"].tolist())
    grid = ssg.Grid([traj1], style = ssg.Gridstyle(x_label="variable 1", y_label = "variable 2"))
    grid.draw()


if __name__ == '__main__':
    test1()
