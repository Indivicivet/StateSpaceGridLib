"""
A number of utility functions acting on data that ideally don't want to appear in the grid interface
"""

import math


def calculate_scale(difference):
    """
    return desired scale
    implemented as the biggest power of 10 smaller than the difference
    """
    scale_factor = 1
    while scale_factor < difference:
        scale_factor *= 10
    while scale_factor > difference:
        scale_factor /= 10
    return scale_factor


def calculate_min_max(values):
    """assumes categorical = string, ordinal = numeric"""
    if isinstance(values[0], str):
        return 0, len(values)
    return int(min(values)), math.ceil(max(values))


def offset_within_bin(
    x_data, x_scale,
    y_data, y_scale,
    bin_counts, visit_count,
):
    """warning: mutates arguments!!"""
    partition_counts = {
        (x, y): 1 << (count - 1).bit_length()
        for (x, y), count in bin_counts.items()
    }
    offset_x = []
    offset_y = []
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        if partition_counts[(x, y)] > 1:
            # todo :: numpy
            direction = (-1, 1)  # todo :: ????
            angle = 2 * math.pi / partition_counts[(x, y)]
            rotation = (
                (math.cos(angle), -math.sin(angle)),
                (math.sin(angle), math.cos(angle)),
            )
            for rotation_number in range(visit_count[(x, y)]):
                # todo :: jesus christ
                direction = (
                    direction[0]*rotation[0][0] + direction[1]*rotation[0][1],
                    direction[0]*rotation[1][0] + direction[1]*rotation[1][1],
                )
            offset_x.append(x + direction[0] * x_scale / 4)
            offset_y.append(y + direction[1] * y_scale / 4)
            visit_count[(x, y)] += 1
    return offset_x, offset_y
    # todo :: should really return x_data', y_data', visit_count' also eh?
