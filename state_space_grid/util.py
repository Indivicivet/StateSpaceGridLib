"""
A number of utility functions acting on data that ideally don't want to appear in the grid interface
"""

import math


def offset_within_bin(
    x_data,
    cell_size_x,
    y_data,
    cell_size_y,
    bin_counts,
    visit_count,  # todo :: we mutate this :(
):
    """
    warning: mutates arguments!!
    in particular, `visit_count`
    """
    partition_counts = {
        (x, y): 1 << (count - 1).bit_length()
        for (x, y), count in bin_counts.items()
    }
    offset_x = []
    offset_y = []
    for x, y in zip(x_data, y_data):
        if partition_counts[(x, y)] > 1:
            # todo :: numpy
            direction = (-1, 1)  # todo :: ????
            angle = 2 * math.pi / partition_counts[(x, y)] * visit_count[(x, y)]
            rotation = (
                (math.cos(angle), -math.sin(angle)),
                (math.sin(angle), math.cos(angle)),
            )
            direction = (
                direction[0]*rotation[0][0] + direction[1]*rotation[0][1],
                direction[0]*rotation[1][0] + direction[1]*rotation[1][1],
            )
            offset_x.append(x + direction[0] * cell_size_x / 4)
            offset_y.append(y + direction[1] * cell_size_y / 4)
        else:
            offset_x.append(x)
            offset_y.append(y)
        visit_count[(x, y)] += 1

    return offset_x, offset_y
    # todo :: should really return x_data', y_data', visit_count' also eh?
