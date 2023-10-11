from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    return gradient_array(shape, from_=0, to=-1, angle=180)


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    return gradient_array(shape, from_=1, to=0, angle=0)


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
        right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    tolerance = np.array([10, 10, 10])
    white_lower_hsv = np.array([0, 0, 166]) - tolerance
    white_upper_hsv = np.array([179, 25, 189]) + tolerance
    yellow_lower_hsv = np.array([26, 84, 176]) - tolerance
    yellow_upper_hsv = np.array([29, 171, 189]) + tolerance

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left_edge = mask_yellow
    mask_right_edge = mask_white

    return mask_left_edge, mask_right_edge

def gradient_array(shape, from_, to, angle):
    # Create a grid of x and y coordinates
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    
    # Create a direction vector from the angle
    direction = [np.cos(np.radians(angle)), np.sin(np.radians(angle))]
    
    # Compute the dot product between each point in the grid and the direction vector
    projection = x * direction[0] + y * direction[1]
    
    # Scale the projection to have values ranging from a to b
    result = from_ + (to - from_) * (projection - projection.min()) / (projection.max() - projection.min())
    
    return result
