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
            # note: this sign convention is arbitrary and for fun
            angle = 2 * math.pi / partition_counts[(x, y)] * visit_count[(x, y)] + 0.75 * math.pi
            sin = math.sin(angle)
            cos = math.cos(angle)
            offset_x.append(x + cos * cell_size_x / 4 * (2**0.5))
            offset_y.append(y + sin * cell_size_y / 4 * (2**0.5))
        else:
            offset_x.append(x)
            offset_y.append(y)
        visit_count[(x, y)] += 1

    return offset_x, offset_y
    # todo :: should really return x_data', y_data', visit_count' also eh?
