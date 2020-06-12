"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Defines the interface of the Display module of the racecar_core library.
"""

import abc
import numpy as np
import math
from typing import List, Tuple, Any
from nptyping import NDArray


class Display(abc.ABC):
    """
    Allows the user to print images to the screen.
    """

    # The radius of the cross used to denote a point in a depth image
    __CROSS_RADIUS = 3

    @abc.abstractmethod
    def show_color_image(self, image: NDArray) -> None:
        """
        Displays a color image in a window.

        Args:
            image: The color image to display to the the screen.

        Example:
            image = rc.camera.get_color_image()

            # Show the image captured by the camera
            rc.display.show_color_image(image)
        """
        pass

    def show_depth_image(
        self,
        image: NDArray[(Any, Any), np.float32],
        max_depth: int = 1000,
        points: List[Tuple[int, int]] = [],
    ) -> None:
        """
        Displays a depth image in grayscale in a window.

        Args:
            image: The depth image to display to the screen.
            max_depth: The farthest depth to show in the image in cm.  Anything past
                this depth is shown as black.
            points: A list of points in (pixel row, pixel column) format to show on
                the image as black crosses.

        Example:
            depth_image = rc.camera.get_depth_image()

            # Show the depth_image captured by the camera.
            rc.display.show_depth_image(depth_image)

            # Show anything that is at most 500 cm away, and show a black cross at
            # row 3, column 5
            rc.display.show_depth_image(depth_image, 500, [(3, 5)])
        """
        assert max_depth > 0, "max_depth must be positive."
        for point in points:
            assert (
                0 <= point[0] < image.shape[0] and 0 <= point[1] < image.shape[1]
            ), "The point {} is not a valid pixel row and column within image.".format(
                point
            )

        # Clip anything above max_depth
        np.clip(image, None, max_depth, image)

        # Shift down 1 unit so that 0 (no data) becomes the "farthest" color
        image = (image - 1) % max_depth

        # Scale measurements so that the closest depth becomes 255 (white) and max_depth
        # becomes 0 (black).
        image = 1 - (image / max_depth)

        # Draw a black plus at each point in points
        for (row, col) in points:
            for i in range(-self.__CROSS_RADIUS, self.__CROSS_RADIUS):
                if 0 <= row + i < image.shape[0]:
                    image[row + i][col] = 0
                if 0 <= col + i < image.shape[1]:
                    image[row][col + i] = 0

        self.show_color_image(image)

    def show_lidar(
        self,
        samples: NDArray[Any, np.float32],
        radius: int = 128,
        max_range: int = 1000,
        highlighted_samples: List[Tuple[float, float]] = [],
    ) -> None:
        """
        Displays a set of LIDAR samples.

        Args:
            samples: A complete LIDAR scan.
            radius: Half of the width or height (in pixels) of the generated image.
            max_range: The farthest depth to show in the image in cm.  Anything past
                this depth is shown as black.
            highlighted_samples: A list of samples in (angle, distance) format to show
                as light blue dots.  Angle must be in degrees from straight ahead
                (clockwise), and distance must be in cm.

        Note:
            Each sample in samples is shown as a red pixel.  Each sample in
            highlighted_samples is shown as a blue pixel.  The car is shown as a green
            dot at the center of the visualization.

        Warning:
            samples must be a complete LIDAR scan.  This function assumes that each
            sample is equal angle appart, and that samples spans the entire 360 degrees.
            If this is not the case, the visualization will be inaccurate.

        Example:
            depth_image = rc.camera.get_depth_image()

            # Show the depth_image captured by the camera.
            rc.display.show_depth_image(depth_image)

            # Show anything that is at most 500 cm away, and show a black cross at
            # row 3, column 5
            rc.display.show_depth_image(depth_image, 500, [(3, 5)])
        """
        assert radius > 0, "radius must be positive."
        assert max_range > 0, "max_range must be positive."

        # Create a square black image with the requested radius
        image = np.zeros((2 * radius, 2 * radius, 3), np.uint8, "C")
        num_samples: int = len(samples)

        # Draw a red pixel for each non-zero sample less than max_range
        for i in range(num_samples):
            if 0 < samples[i] < max_range:
                angle: float = 2 * math.pi * i / num_samples
                length: float = radius * samples[i] / max_range
                r: int = int(radius - length * math.cos(angle))
                c: int = int(radius + length * math.sin(angle))
                image[r][c][2] = 255

        # Draw a green dot to denote the car
        for r in range(radius - 1, radius + 1):
            for c in range(radius - 1, radius + 1):
                image[r][c][1] = 255

        # Draw a light blue pixel for each point in highlighted_samples
        for (angle, distance) in highlighted_samples:
            if 0 < distance < max_range:
                angle_rad = angle * math.pi / 180
                length: float = radius * distance / max_range
                r: int = int(radius - length * math.cos(angle_rad))
                c: int = int(radius + length * math.sin(angle_rad))
                image[r][c][0] = 255
                image[r][c][1] = 255
                image[r][c][2] = 0

        self.show_color_image(image)
