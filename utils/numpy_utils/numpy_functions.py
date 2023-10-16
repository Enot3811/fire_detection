"""The module that contains some functions manipulating with numpy data."""


from typing import List, Tuple

import numpy as np


def rotate_rectangle(
    points: List[Tuple[int, int]], angle: float, radians: bool = True
) -> List[Tuple[int, int]]:
    """Rotate points of rectangle by given angle.

    Parameters
    ----------
    points : List[Tuple[int, int]]
        Points of rectangle. There must be 2 or 4.
    angle : float
        Angle to rotate.
    radians : bool, optional
        Whether angle given in radians. Otherwise in degrees. By default equals
        `True` (radians).

    Returns
    -------
    List[Tuple[int, int]]
        List of rotated rectangle points.
    """
    if not radians:
        angle = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)

    points = [
        (
            int(c_x + np.cos(angle) * (px - c_x) - np.sin(angle) * (py - c_y)),
            int(c_y + np.sin(angle) * (px - c_x) + np.cos(angle) * (py - c_y)))
        for px, py in points]
    return points
