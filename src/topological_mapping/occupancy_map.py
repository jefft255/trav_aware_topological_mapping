import numpy as np
from math import ceil, floor
from typing import Tuple
from scipy.signal import convolve2d
import rospy
import numpy.typing as npt


class OccupancyMap:
    def __init__(self, min: float, max: float, res: float) -> None:
        self.min = min
        self.max = max
        self.max += float(res) / 2.0
        self.min -= float(res) / 2.0
        self.n_cells_x = ceil((float(self.max) - float(self.min)) / float(res))
        self.res = res
        self.buffer = np.zeros((self.n_cells_x, self.n_cells_x), dtype=np.uint8)

    def get_buffer_coordinates(
        self, coordinates: npt.NDArray[np.float32]
    ) -> Tuple[int, int]:
        """Get buffer index from float coordinates

        Args:
            coordinates (np.ndarray): x, y in global coordinates

        Returns:
            Tuple[int, int]: index in buffer
        """
        x, y = coordinates

        x = max(min(x, self.max), self.min)
        y = max(min(y, self.max), self.min)

        x -= self.min
        y -= self.min
        x /= self.res
        y /= self.res
        x, y = int(floor(x)), int(floor(y))

        x = min(x, self.buffer.shape[0] - 1)
        y = min(y, self.buffer.shape[1] - 1)
        return x, y

    def get_global_coordinates(self, buffer_coordinates: Tuple[int, int]) -> npt.NDArray[np.float32]:
        """Get global coordinates from buffer index

        Args:
            buffer_coordinates (Tuple[int, int]): index in buffer

        Returns:
            np.ndarray: x, y in global coordinates
        """
        x, y = buffer_coordinates

        assert x in range(self.buffer.shape[0])
        assert y in range(self.buffer.shape[1])

        x *= self.res
        y *= self.res
        x += self.min + float(self.res) / 2.0
        y += self.min + float(self.res) / 2.0
        return np.array([x, y])

    def add_visit(self, coordinates: npt.NDArray[np.float32]) -> None:
        self.buffer[self.get_buffer_coordinates(coordinates)] = 1

    def compute_frontiers(self) -> None:
        self.buffer[self.buffer != 1] = 0
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        has_visited_neighbords = convolve2d(self.buffer, kernel, mode="same")
        is_frontier = np.logical_and(self.buffer == 0, has_visited_neighbords > 0)
        self.buffer[is_frontier] = 2
        # rospy.logwarn(f"Frontiers: {np.argwhere(self.buffer == 2)}")
        # rospy.logwarn(f"Frontiers: {np.argwhere(self.buffer == 1)}")

    def get_frontiers(self) -> npt.NDArray[np.float32]:
        indexes = self.get_frontiers_index()
        global_coordinates = np.zeros(indexes.shape, dtype=np.float32)
        for i in range(indexes.shape[0]):
            global_coordinates[i] = self.get_global_coordinates(indexes[i])
        return global_coordinates

    def get_frontiers_index(self) -> npt.NDArray[np.intp]:
        return np.argwhere(self.buffer == 2)


if __name__ == "__main__":
    map = OccupancyMap(-5, 5, 1.0)
    assert map.buffer.shape == (10, 10)
    map.add_visit(np.array([-5, -5]))
    assert map.buffer[0, 0] == 1
    map.add_visit(np.array([5, 5]))
    assert map.buffer[4, 4] == 1
    print(map.buffer)
    map.compute_frontiers()
    print(map.buffer)
    map.add_visit(np.array([-5, -5]))
    map.compute_frontiers()
    print(map.buffer)
    print(map.get_global_coordinates(map.get_buffer_coordinates(np.array([0.5, 0.5]))))
