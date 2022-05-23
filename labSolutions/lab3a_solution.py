"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3A - Depth Camera Safety Stop
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
MIN_STOP_DISTANCE = 20  # cm
MAX_STOP_DISTANCE = 80  # cm
ALPHA = 0.2  # Amount to use current_speed when updating cur_speed

# Boundaries of the crop window used to only consider objects in front of the car
LEFT_COL = int(rc.camera.get_width() * 0.3)
RIGHT_COL = int(rc.camera.get_width() * 0.6)
BOTTOM_ROW = int(rc.camera.get_height() * 0.65)

# Amount to increase stop distance (cm) per speed (cm/s) squared
STOP_DISTANCE_SCALE = 40.0 / 10000

# slow_distance / stop_distance
SLOW_DISTANCE_RATIO = 1.5

# >> Variables
cur_speed = 0  # cm/s
prev_distance = 0  # cm


########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_speed
    global prev_distance

    # Have the car begin at a stop
    rc.drive.stop()

    # Initialize variables
    cur_speed = 0
    depth_image = rc.camera.get_depth_image()
    prev_distance = rc_utils.get_depth_image_center_distance(depth_image)

    # Print start message
    print(
        ">> Lab 3A - Depth Camera Safety Stop\n"
        "\n"
        "Controls:\n"
        "   Right trigger = accelerate forward\n"
        "   Right bumper = override safety stop\n"
        "   Left trigger = accelerate backward\n"
        "   Left joystick = turn front wheels\n"
        "   A button = print current speed and angle\n"
        "   B button = print the distance at the center of the depth image"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_speed
    global prev_distance

    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Calculate the distance of the object directly in front of the car by cropping
    # out a window directly in front of the car and finding the closest point
    depth_image = rc.camera.get_depth_image()
    depth_image_cropped = rc_utils.crop(
        depth_image, (0, LEFT_COL), (BOTTOM_ROW, RIGHT_COL)
    )
    closest_point = rc_utils.get_closest_pixel(depth_image_cropped)
    distance = rc_utils.get_pixel_average_distance(depth_image_cropped, closest_point)

    # Update forward speed estimate
    frame_speed = (prev_distance - distance) / rc.get_delta_time()
    cur_speed += ALPHA * (frame_speed - cur_speed)
    prev_distance = distance

    # Calculate slow and stop distances based on the forward speed
    stop_distance = rc_utils.clamp(
        MIN_STOP_DISTANCE + cur_speed * abs(cur_speed) * STOP_DISTANCE_SCALE,
        MIN_STOP_DISTANCE,
        MAX_STOP_DISTANCE,
    )
    slow_distance = stop_distance * SLOW_DISTANCE_RATIO

    if not rc.controller.is_down(rc.controller.Button.RB) and cur_speed > 0:
        # If we are past slow_distance, reduce speed proportional to how close we are
        # to stop_distance
        if stop_distance < distance < slow_distance:
            speed = min(
                speed,
                rc_utils.remap_range(distance, stop_distance, slow_distance, 0, 0.5),
            )
            print("Safety slow: speed limited to {}".format(speed))

        # Safety stop if we are passed stop_distance by reversing at a speed
        # proportional to how far we are past stop_distance
        if 0 < distance < stop_distance:
            speed = rc_utils.remap_range(distance, 0, stop_distance, -4, -0.2, True)
            speed = rc_utils.clamp(speed, -1, -0.2)
            print("Safety stop: reversing at {}".format(speed))

    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the depth image closest distance when the B button is held down
    if rc.controller.is_down(rc.controller.Button.B):
        print("Distance:", distance)

    # Print cur_speed estimate and stop distance when the X button is held down
    if rc.controller.is_down(rc.controller.Button.X):
        print(
            "Current speed estimate: {:.2f} cm/s, Stop distance: {:.2f}".format(
                cur_speed, stop_distance
            )
        )

    # Display the current depth image
    rc.display.show_depth_image(
        depth_image, points=[(closest_point[0], closest_point[1] + LEFT_COL)]
    )

    # TODO (stretch goal): Prevent forward movement if the car is about to drive off a
    # ledge.  ONLY TEST THIS IN THE SIMULATION, DO NOT TEST THIS WITH A REAL CAR.


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
