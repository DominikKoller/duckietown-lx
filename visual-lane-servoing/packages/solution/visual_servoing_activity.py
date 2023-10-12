from typing import Tuple

import numpy as np
import cv2

gain = 0.06
gain_yellow = 1.6
far_intensity = 0.0

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    result = gradient_array(shape, from_=-0.5, to=-1, angle=0)

    result = result * gradient_array(shape, from_=far_intensity, to=1, angle=90)

    return result * gain * gain_yellow


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    rows, cols = shape
    gradient_start_col = cols // 3
    result = np.zeros(shape)
    
    gradient_shape = (rows, cols - gradient_start_col)
    gradient_part = gradient_array(gradient_shape, from_=1, to=0.5, angle=0)
    
    result[:, gradient_start_col:] = gradient_part

    result = result * gradient_array(shape, from_=far_intensity, to=1, angle=90)

    return result * gain


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
        right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    tolerance = np.array([20, 20, 20])
    white_lower_hsv = np.array([0, 0, 166]) - tolerance
    white_upper_hsv = np.array([179, 25, 189]) + tolerance
    yellow_lower_hsv = np.array([26, 84, 176]) - tolerance
    yellow_upper_hsv = np.array([29, 240, 200]) + tolerance

    masked_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    masked_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    return lane_markings_from_masked(masked_yellow), lane_markings_from_masked(masked_white)

def lane_markings_from_masked(image):
    sigma = 1
    image = cv2.GaussianBlur(image,(0,0), sigma)

    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1)

     # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    return Gmag # (Gmag > 20)

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
