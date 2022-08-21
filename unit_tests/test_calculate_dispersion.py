import pytest
from state_space_grid import grid, trajectory

@pytest.mark.parametrize("grid_points", [2, 5])  # nb. 1 == div/0
def test_calculate_dispersion_single_point(grid_points):
    traj = trajectory.Trajectory(
        data_x=[1],
        data_y=[1],
        data_t=[0, 1],
    )
    result = grid.calculate_dispersion([traj], grid_points)
    assert result == 0
