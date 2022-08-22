"""
A number of utility functions acting on data that ideally don't want to appear in the grid interface
"""

import math
import cmath


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
        pos_cx = (
            cmath.exp(
                # note: this sign convention is arbitrary and for fun
                1j * math.pi * (2 / partition_counts[(x, y)] * visit_count[(x, y)] + 0.75)
            ) * 2 ** 0.5 / 4
            if partition_counts[(x, y)] > 1
            else 0
        )
        offset_x.append(x + pos_cx.real * cell_size_x)
        offset_y.append(y + pos_cx.imag * cell_size_y)
        visit_count[(x, y)] += 1

    return offset_x, offset_y
    # todo :: should really return x_data', y_data', visit_count' also eh?
