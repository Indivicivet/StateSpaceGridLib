from pathlib import Path

from state_space_grid import trajectory, grid

DATA_SRC = Path(__file__).parent / "resources"

FILES = list(DATA_SRC.glob("*.trj"))
for file in FILES:
    print(file.name)

g = grid.Grid(
    [
        trajectory.Trajectory.from_legacy_trj(path)
        for path in FILES
    ]
)

print(g.get_measures())
