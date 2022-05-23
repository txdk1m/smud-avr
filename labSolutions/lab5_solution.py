"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5 - AR Markers
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import enum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()


class Stage(enum.IntEnum):
    none = 0
    marker_id = 1
    orientation = 2
    color = 3


# >> Constants
RED = ((170, 50, 50), (10, 255, 255), "red")  # The HSV range for the color blue
BLUE = ((100, 150, 150), (120, 255, 255), "blue")  # The HSV range for the color blue
GREEN = ((40, 50, 50), (80, 255, 255), "green")  # The HSV range for the color green

# Wall follow constants
# Angle to the right side of the car
SIDE_ANGLE = 90

# Spread to look in each direction to the side of the car to estimate wall angle
SPREAD_ANGLE = 45

# The angle of measurements to average over for each distance measurement
WINDOW_ANGLE = 5

# Distance in (in cm) to try stay from the wall
WALL_DISTANCE = 35

# The maximum difference between the desired value and the current value, which is
# scaled to a full left (-1) or a full right (1) turn
MAX_DIF = 10

# The amount we consider the distance from the wall compared to the angle of the wall
DISTANCE_COEFFICIENT = 0.5

# Line follow constants
# Minimum number of pixels to consider a valid contour
MIN_CONTOUR_AREA = 30

# Region to which to crop the image when looking for the colored line
CROP_WINDOW = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Speed constants
EXPLORE_SPEED = 0.4
WALL_FOLLOW_SPEED = 0.5
LINE_FOLLOW_SPEED = 0.7

# >> Variables
cur_stage = Stage.none

# The most recent marker detected (initialized to a non-existant marker)
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))

# -1 if we are following the left wall, 1 if we are following the right wall
cur_direction = 0

# The color of the line we are currently following
cur_color = None

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color

    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 5 - AR Markers")

    # Initialize variables
    cur_stage = Stage.none
    cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))
    cur_direction = 0
    cur_color = None


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color

    color_image = rc.camera.get_color_image_no_copy()
    markers = rc_utils.get_ar_markers(color_image)

    # If we see a new marker, change the stage
    if len(markers) > 0 and markers[0].get_id() != cur_marker.get_id():
        change_stage(markers[0], color_image)

    speed = 0
    angle = 0
    if cur_stage == Stage.none:
        # Until we see the first marker, gradually move forward
        speed = EXPLORE_SPEED
        angle = 0
    elif cur_stage == Stage.marker_id or cur_stage == Stage.orientation:
        # After the first two markers, follow the wall indicated by the marker
        speed = WALL_FOLLOW_SPEED
        angle = wall_follow(cur_direction)
    elif cur_stage == Stage.color:
        # After the third marker, follow the color line indicated by the marker
        speed = LINE_FOLLOW_SPEED
        angle = line_follow([cur_color, GREEN], color_image)

        # If we cannot see the colored line yet, continue wall following
        if angle is None:
            angle = wall_follow(cur_direction)

    rc.drive.set_speed_angle(speed, angle)

    # Print global variables when the X button is held down
    if rc.controller.is_down(rc.controller.Button.X):
        print(
            f"cur_stage: {cur_stage}, cur_direction: {cur_direction}, cur_color: {cur_color}"
        )


def change_stage(new_marker, color_image):
    """
    Moves to the next stage when a new marker is detected.
    """
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color

    cur_stage += 1
    cur_marker = new_marker

    if cur_stage == Stage.marker_id:
        cur_direction = 1 if cur_marker.get_id() == 1 else -1
    elif cur_stage == Stage.orientation:
        cur_direction = (
            1 if cur_marker.get_orientation() == rc_utils.Orientation.RIGHT else -1
        )
    elif cur_stage == Stage.color:
        cur_marker.detect_colors(color_image, [RED, BLUE])
        cur_color = RED if new_marker.get_color() == "red" else BLUE


def wall_follow(direction):
    """
    Determines the angle of the wheels necessary to follow the wall on the side
    specified by the direction parameter.

    Uses a similar strategy to Lab 5B.
    """
    lidar_scan = rc.lidar.get_samples()

    # Measure 3 points along the wall we are trying to follow
    side_dist = side_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_ANGLE * direction, WINDOW_ANGLE
    )
    side_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, (SIDE_ANGLE - SPREAD_ANGLE) * direction, WINDOW_ANGLE
    )
    side_back_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, (SIDE_ANGLE + SPREAD_ANGLE) * direction, WINDOW_ANGLE
    )

    # Determine a goal angle based on how we are aligned with the wall
    dif_component = rc_utils.remap_range(
        side_front_dist - side_back_dist, -MAX_DIF, MAX_DIF, -1, 1, True
    )

    # Determine a goal angle based on how far we are from the wall
    distance_component = rc_utils.remap_range(
        WALL_DISTANCE - side_dist, -MAX_DIF, MAX_DIF, 1, -1, True
    )

    # Take a linear combination of the two goal angles
    angle = dif_component + distance_component * DISTANCE_COEFFICIENT
    return direction * rc_utils.clamp(angle, -1, 1)


def line_follow(colors, color_image):
    """
    Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A.
    """
    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_WINDOW[0], CROP_WINDOW[1])

    # Search for the colors in priority order
    for color in colors:
        # Find the largest contour of the specified color
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour is not None:
            center = rc_utils.get_contour_center(contour)
            return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    # If no color lines are found, return None so that we wall follow instead
    return None


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
