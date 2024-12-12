import numpy as np
import pickle
import math
import cv2
from pathlib import Path
from topological_mapping.topological_map import RealMapNode, MapNode


# load instrinsic matrices for the three cameras

front_intrinsic = np.array(
    [
        [2.41394434e03, 0.00000000e00, 2.03381238e03],
        [0.00000000e00, 2.41394434e03, 1.51682800e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

left_intrinsic = np.array(
    [
        [2.42406909e03, 0.00000000e00, 2.02071301e03],
        [0.00000000e00, 2.42406909e03, 1.51757178e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

right_intrinsic = np.array(
    [
        [2.41342725e03, 0.00000000e00, 2.06299438e03],
        [0.00000000e00, 2.41342725e03, 1.57475842e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# right camera rotation matrix is Rz(pi).Rx(pi/2)
rot_right_cam = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

# front camera rotation matrix is Rz(-pi/2).Ry(pi/2)
rot_front_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

# left camera rotation matrix is Rx(-pi/2)
rot_left_cam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

c_right_cam = np.array([[0.0], [-0.25], [0.0]])
c_front_cam = np.array([[0.2985], [0.0], [0.0]])
c_left_cam = np.array([[0.0], [0.25], [0.0]])

ric = lambda rot, c: np.matmul(rot, np.hstack((np.eye(3), -c)))  # R[I|-C]
fpc = lambda k, rot, c: np.matmul(k, ric(rot, c))  # P = K R [I | -C]


def project_node(nodeA: RealMapNode, nodeB: MapNode, color=None):
    # cam1, _ = pick_cam(nodeA, nodeB)
    for cam_name, K, R, C, img in zip(
        ["front", "left", "right"],
        [front_intrinsic, left_intrinsic, right_intrinsic],
        [rot_front_cam, rot_left_cam, rot_right_cam],
        [c_front_cam, c_left_cam, c_right_cam],
        [nodeA.image_f, nodeA.image_l, nodeA.image_r],
    ):
        # img = cv2.resize(img, (4056, 3040), interpolation=cv2.INTER_AREA)
        original_intrinsic_width = 4056
        current_width = img.shape[1]
        ratio = float(current_width) / float(original_intrinsic_width)
        K = ratio * K
        K[2, 2] = 1.0

        G2A_translation = np.append(
            nodeA.translation, 1
        )  # translation of node 2 w.r.t node 1
        G2A_transformation_matrix = np.zeros((4, 4))
        G2A_transformation_matrix[:3, :3] = nodeA.rotation[:3, :3]
        G2A_transformation_matrix[:, 3] = G2A_translation

        nodeB_wrtA = np.matmul(
            np.linalg.inv(G2A_transformation_matrix), np.append(nodeB.translation, 1)
        )  # if just need to rotate to A

        P = fpc(K, R, C)  # finite projective camera model
        P_3d = ric(R, C) @ nodeB_wrtA  # Node B is A's camera frame!
        pixel_point = np.matmul(P, nodeB_wrtA)  # getting homogeneous pixel coordinates
        pixel_point = (
            pixel_point[:2] / pixel_point[2]
        )  # removing homogeneous coordinates
        # print(f"{cam_name} camera")
        # print(pixel_point)
        # Yes this is confusing but pixel_point is x,y and img.shape is height, width
        if (0 < pixel_point[0] < img.shape[1]) and (0 < pixel_point[1] < img.shape[0]):
            if P_3d[2] > 0:  # Otherwise point is *behind* camera
                if color == "blue":
                    color = (255, 0, 0)
                elif color == "yellow":
                    color = (0, 255, 255)
                # print(pixel_point)
                if color is not None:
                    img = cv2.circle(
                        img.copy(),
                        (int(pixel_point[0]), int(pixel_point[1])),
                        20,
                        color,
                        20,
                    )
                return img, cam_name, P_3d, pixel_point
    return None, None, None, None
