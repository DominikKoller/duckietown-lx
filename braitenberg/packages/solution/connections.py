from typing import Tuple

import numpy as np

attraction_top = 1
attraction_bottom = 0.5
repulsion_top = 0.5
repulsion_bottom = 1


repulsion_bottom_min = 0.2

top_bottom_cutoff = 0.45

left_bias = 0.5

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    rows = res.shape[0]
    cols = res.shape[1]
    
    #for i in range(cols):
    #    res[rows//2:, i] = 1 if i- (2 * (i / cols) - 1)

    cutoff = round(rows*top_bottom_cutoff)
    vertical_cutoff = round(cols*left_bias)

    # res[:cutoff, :vertical_cutoff] = repulsion_top
    # res[cutoff:, :cols//2] = repulsion_bottom

    
    
    #for i in range(vertical_cutoff):
    #    res[:cutoff, i] = (1 - i / (vertical_cutoff)) * repulsion_top

    #for i in range(cols//2):
    #    res[cutoff:, i] = max(repulsion_bottom_min, repulsion_bottom * (i/(cols//2)))

    
    res[:cutoff, :vertical_cutoff] = - attraction_top # top left
    res[:cutoff, vertical_cutoff:] = repulsion_top # top right
    res[cutoff:, :vertical_cutoff] = repulsion_bottom
    res[cutoff:, vertical_cutoff:] = - attraction_bottom 

    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    rows = res.shape[0]
    cols = res.shape[1]

    cutoff = round(rows*top_bottom_cutoff)
    vertical_cutoff = round(cols*left_bias)

    # res[:cutoff, :vertical_cutoff] = repulsion_top
    # res[cutoff:, cols//2:] = repulsion_bottom

    #for i in range(vertical_cutoff, cols):
    #    p = (i-vertical_cutoff) / (cols - vertical_cutoff)
    #    res[:cutoff, i] = p * repulsion_top

    #for i in range(cols//2, cols):
    #    p = (i-cols//2) / (cols - (cols//2))
    #    res[cutoff:, i] = max(repulsion_bottom_min, repulsion_bottom * (1-p))

    res[:cutoff, :vertical_cutoff] = repulsion_top # top left
    res[:cutoff, vertical_cutoff:] = - attraction_top # top right
    res[cutoff:, :vertical_cutoff] = - attraction_bottom
    res[cutoff:, vertical_cutoff:] = repulsion_bottom

    return res
