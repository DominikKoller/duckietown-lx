from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        rotation_wheel: Rotation of the wheel in radians.
        ticks: current number of ticks.
    """

    alpha = 2 * np.pi / resolution # wheel rotation per tick in radians

    delta_ticks = ticks - prev_ticks
    dphi = 2 * np.pi * delta_ticks / resolution

    return dphi, ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x:                  estimated x coordinate
        y:                  estimated y coordinate
        theta:              estimated heading
    """

    L = baseline / 2

    try:
        delta_q = (R/2) * np.array([
                [np.cos(theta_prev), 0],
                [np.sin(theta_prev), 0],
                [0, 1]
            ]) @ np.array([
                [1, 1],
                [1/L, -1/L]
            ]) @ np.array([
                [delta_phi_right],
                [delta_phi_left]
            ])
    except:
        print("exception")
        print("R: " + str(R))
        print("baseline: " + str(baseline))
        print("x_prev: " + str(x_prev))
        print("y_prev: " + str(y_prev))
        print("theta_prev: " + str(theta_prev))
        print("delta_phi_left: " + str(delta_phi_left))
        print("delta_phi_right: " + str(delta_phi_right))

    # These are random values, replace with your own
    x_curr = x_prev + delta_q[0][0]
    y_curr = y_prev + delta_q[1][0]
    theta_curr = theta_prev + delta_q[2][0]
    # ---
    return x_curr, y_curr, theta_curr
