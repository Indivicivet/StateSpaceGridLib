import state_space_grid as ssg
import pandas as pd
import os

def test1():
    file_path = os.path.dirname(__file__)
    data1 = pd.read_csv(open((file_path+"\\" if file_path else "") + r"..\resources\ExampleData1.txt"))
    traj1 = ssg.Trajectory(data1["variable 1"].tolist(),data1["variable 2"].tolist(), data1["Onset"].tolist())
    grid = ssg.Grid([traj1], style=ssg.GridStyle(x_label="variable 1", y_label ="variable 2", ))
    grid.draw()

def test2():
    file_path = os.path.dirname(__file__)
    data1 = pd.read_csv(open((file_path+"\\" if file_path else "") + r"..\resources\ExampleData1.txt"))
    traj1 = ssg.Trajectory(data1["variable 1"].tolist(), data1["variable 2"].tolist(), data1["Onset"].tolist())
    #traj1.addYOrdering(["Low", "Medium", "High"])
    grid = ssg.Grid([traj1], style=ssg.GridStyle(title="test 2", x_label="variable 1", y_label ="variable 2", title_font_size=30, x_max=4, x_min=1, y_max=4, y_min=1))
    #grid.draw()
    measure = grid.get_measures()
    print(measure)

if __name__ == '__main__':
    test2()
