"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2021

Example Lane Following Solution
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 500

# A crop window for the floor directly in front of the car
CROP_FLOOR = (
    (rc.camera.get_height() * 2 // 3, 0),
    (rc.camera.get_height(), rc.camera.get_width()),
)

# Possible lane colors, stored as (hsv_min, hsv_max, name)
ORANGE = ((10, 100, 100), (25, 255, 255), "orange")
PURPLE = ((125, 100, 100), (150, 255, 255), "purple")

# ID of the AR marker which gives the lane primary color
AR_MARKER_ID = 1

MAX_SPEED = 0.35
# Speed to use when the lane is the primary color
FAST_SPEED = 1.0
# Speed to use when the lane is the secondary color (indicating a sharp turn)
SLOW_SPEED = 0.5

# Amount to turn if we only see one lane
ONE_LANE_TURN_ANGLE = 0.75

# >> Variables
# True if the AR marker has been seen, meaning we can begin driving
driving = False
# primary lane color (found on AR tag)
primary_color = ORANGE
# secondary lane color, indicating a sharp turns
secondary_color = PURPLE

########################################################################################
# Functions
########################################################################################


def check_ar():
    """
    Looks for an AR marker indicating that we should begin lane following, and updates
    the primary color to the color of the AR marker
    """
    global driving
    global primary_color
    global secondary_color

    image = rc.camera.get_color_image_no_copy()
    ar_markers = rc_utils.get_ar_markers(image, [ORANGE, PURPLE])

    for marker in ar_markers:
        if marker.get_id() == AR_MARKER_ID:
            if not driving:
                print("AR marker found, begin driving")
                driving = True

            # Set the primary color to the color of the marker
            if marker.get_color() != primary_color[2]:
                temp = primary_color
                primary_color = secondary_color
                secondary_color = temp
                print(f"Primary color set to {primary_color[2]}")


def start():
    """
    This function is run once every time the start button is pressed
    """
    global driving
    global primary_color
    global secondary_color

    # Initialize variables
    driving = False
    primary_color = ORANGE
    secondary_color = PURPLE

    # Set max speed
    rc.drive.set_max_speed(MAX_SPEED)

    # Begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Sample Lane Following Solution")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    speed = SLOW_SPEED
    angle = 0

    check_ar()

    if not driving:
        # No AR marker seen yet
        rc.drive.stop()
        return

    image = rc.camera.get_color_image()
    if image is None:
        print("No image")
        rc.drive.stop()
        return

    # Crop the image to the floor directly in front of the car
    image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

    # Search for secondary (slow) color first
    contours = [
        contour
        for contour in rc_utils.find_contours(
            image, secondary_color[0], secondary_color[1]
        )
        if cv.contourArea(contour) > MIN_CONTOUR_AREA
    ]

    if len(contours) == 0:
        # Secondary color not found, search for primary (fast) color
        contours = [
            contour
            for contour in rc_utils.find_contours(
                image, primary_color[0], primary_color[1]
            )
            if cv.contourArea(contour) > MIN_CONTOUR_AREA
        ]
        if len(contours) == 0:
            # No contours of either color found, so proceed forward slowly
            print("No lanes found")
            rc.drive.set_speed_angle(SLOW_SPEED, 0)
            return
        else:
            # We only see the primary color, so it is safe to go fast
            speed = FAST_SPEED

    # If we see at least two contours, aim for the midpoint of the centers of the two
    # largest contours (assumed to be the left and right lanes)
    if len(contours) >= 2:
        # Sort contours from largest to smallest
        contours.sort(key=cv.contourArea, reverse=True)

        # Calculate the midpoint of the two largest contours
        first_center = rc_utils.get_contour_center(contours[0])
        second_center = rc_utils.get_contour_center(contours[1])
        midpoint = (first_center[1] + second_center[1]) / 2

        # Use P-control to aim for the midpoint
        angle = rc_utils.remap_range(midpoint, 0, rc.camera.get_width(), -1, 1)

        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(image, contours[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(image, contours[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(image, second_center, rc_utils.ColorBGR.blue.value)

    # If we see only one contour, turn back towards the "missing" line
    else:
        contour = contours[0]
        center = rc_utils.get_contour_center(contour)

        if center[1] > rc.camera.get_width() / 2:
            # We can only see the RIGHT lane, so turn LEFT
            angle = -ONE_LANE_TURN_ANGLE
        else:
            # We can only see the LEFT lane, so turn RIGHT
            angle = ONE_LANE_TURN_ANGLE

        # Draw the single contour and center onto the image
        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, center)

    # Display the image to the screen
    rc.display.show_color_image(image)

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
