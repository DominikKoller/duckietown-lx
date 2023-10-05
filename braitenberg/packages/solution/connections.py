from typing import Tuple

import numpy as np

horizon = 0.09
repulsion = 0.3
repulsion_size_top = 0
repulsion_size_bottom = 0.55

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    rows = res.shape[0]
    cols = res.shape[1]

    middle = cols // 2

    horizon_row = round(horizon*rows)

    for i in range(horizon_row, rows):
        p = (i-horizon_row) / (rows - horizon_row)
        repulsion_size = p * (repulsion_size_bottom - repulsion_size_top) + repulsion_size_top
        repulsion_until = round(middle + cols * repulsion_size)
        res[i, middle:repulsion_until] = - repulsion

    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    return np.flip(get_motor_left_matrix(shape), axis=1)
