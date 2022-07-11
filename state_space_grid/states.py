# A number of utility functions acting on data that ideally don't want to appear in the grid interface

import math


def calculate_scale(var_min, var_max):
    scale_factor = 1
    scale = var_max - var_min
    if scale_factor < scale:
        while (scale_factor * 10) < scale:
            scale_factor = scale_factor * 10
    if scale_factor > scale:
        while scale_factor > scale:
            scale_factor = scale_factor / 10
    return scale_factor


def calculate_min_max(vars):
    var_min, var_max = 0, 0
    # assume categorical = string, ordinal = numeric
    if type(vars[0]) == str:
        var_max = len(vars)
    else:
        var_min = int(min(vars))
        var_max = int(max(vars)) + (1 if int(max(vars)) < max(vars) else 0)
    return var_min, var_max


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

