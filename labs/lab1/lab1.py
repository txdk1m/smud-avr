"""
Copyright Harvey Mudd College
Fall 2019

Lab <n> - <Lab Title>
"""

################################################################################
# Imports
################################################################################

import sys
sys.path.insert(0, '../../library')
from racecar_core import *
rospy.init_node('racecar')


################################################################################
# Global variables
################################################################################

rc = Racecar()
counter = 0


################################################################################
# Functions
################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    pass

def update():
    """
    After start() is run, this function is run every frame until the back button is
    pressed
    """
    global counter
    print(counter)
    if (counter < 120):
        rc.drive.set_speed_angle(1, 20)
    else:
        rc.drive.set_speed_angle(0, -10)
    counter += 1


################################################################################
# Do not modify any code beyond this point
################################################################################

if __name__ == "__main__":
    rc.run(start, update)
