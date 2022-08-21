import pytest
from state_space_grid import grid, trajectory

@pytest.mark.parametrize("x0", [0, 1, 3.7])
@pytest.mark.parametrize("y0", [0, 1, 3.7])
@pytest.mark.parametrize("t1", [1, 3.7])
@pytest.mark.parametrize("grid_points", [2, 5])  # nb. 1 == div/0
def test_calculate_dispersion_single_point(x0, y0, t1, grid_points):
    traj = trajectory.Trajectory(
        data_x=[x0],
        data_y=[y0],
        data_t=[0, t1],
    )
    result = grid.calculate_dispersion([traj], grid_points)
    assert result == 0


def test_calculate_dispersion_two_points():
    traj = trajectory.Trajectory(
        data_x=[1, 5],
        data_y=[1, 5],
        data_t=[0, 1, 3],
    )
    result = grid.calculate_dispersion([traj], 99)
    expected = 0.50510204
    assert abs(result - expected) < 1e-3, f"got {result}, expected about {expected}"
