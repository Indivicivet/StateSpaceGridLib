from state_space_grid import grid, trajectory


def test_calculate_dispersion_single_point():
    traj = trajectory.Trajectory(
        data_x=[1],
        data_y=[1],
        data_t=[0, 1],
    )
    result = grid.calculate_dispersion([traj], 5)
    assert result == 0
