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


def offset_within_bin(x_data, x_scale, y_data, y_scale, bin_counts, visit_count):

    partition_counts = {y: {x: (1 << (counts-1).bit_length()) for x, counts in x_and_counts.items()} for y, x_and_counts in bin_counts.items()}
    for i in range(len(x_data)):
        x, y = x_data[i], y_data[i]
        partitions = partition_counts[y][x]
        if partitions > 1:
            direction = (-1, 1)
            rotation = ((math.cos(2*math.pi/partitions), -1*math.sin(2*math.pi/partitions)),
                        (math.sin(2*math.pi/partitions), math.cos(2*math.pi/partitions)))
            for rotation_number in range(visit_count[y][x]):
                direction = (direction[0]*rotation[0][0] + direction[1]*rotation[0][1],
                             direction[0]*rotation[1][0] + direction[1]*rotation[1][1])
            x_data[i] = x+(direction[0]*x_scale/4)
            y_data[i] = y+(direction[1]*y_scale/4)
            visit_count[y][x] += 1

