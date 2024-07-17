from typing import Tuple


def DT_TOKEN() -> str:
    # TODO: change this to your duckietown token
    dt_token = "your token here"
    return dt_token


def MODEL_NAME() -> str:
    # TODO: change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5n"
    return "yolov5n"


def NUMBER_FRAMES_SKIPPED() -> int:
    # TODO: change this number to drop more frames
    # (must be a positive integer)
    return 1


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    """
    # Right now, this returns True for every object's class
    # TODO: Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    return (pred_class == 0)


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    # Right now, this returns True for every object's confidence
    # TODO: Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    return (score > 0.9)


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """

    left, top, right, bottom = bbox

    screen_height = 416
    screen_width = 416
    
    # Condition 1: Discard very small boxes
    min_width = 10  # Example threshold, adjust as needed
    min_height = 10  # Example threshold, adjust as needed
    if (right - left) < min_width or (bottom - top) < min_height:
        return False
    
    # Condition 2: Discard if the lower edge is NOT within the lower middle part of the screen
    lower_part_start_y = 0.55 * screen_height
    middle_part_start_x = 0.1 * screen_width
    middle_part_end_x = 0.9 * screen_width
    
    if bottom < lower_part_start_y:
        return False
    if right < middle_part_start_x or left > middle_part_end_x:
        return False
    
    return True
