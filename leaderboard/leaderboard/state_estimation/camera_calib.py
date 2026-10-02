import math
import numpy as np
import torch
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

CAM_ORDER = ["cam_00", "cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]
CAM_RIG = {
    "cam_00": (0.8, -0.3, -0.18, 0.0),
    "cam_01": (0.8,  0.3, -0.18, 0.0),
    "cam_02": (0.0, -0.46, -0.18, -90.0),
    "cam_03": (0.0,  0.46, -0.18, 90.0),
    "cam_04": (-0.8, -0.3, -0.18, 180.0),
    "cam_05": (-0.8, 0.3, -0.18, 180.0),
}
SRC_W, SRC_H, FOV_DEG = 1408, 376, 103.744808   # raw CARLA camera
OUT_W, OUT_H = 704, 192                          # after cv2.resize in SensorDataParser

# camera axes (right, down, forward) expressed in the lidar frame (fwd, left, up)
_R_CAM_TO_LIDAR = np.array([[0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def build_camera_calibration(lidar2ego: np.ndarray = None) -> dict:
    """
    lidar2ego: optional 4x4 (right-handed lidar frame -> right-handed ego frame).
               Only used for 'camera2ego'; identity if omitted.
    Returns float32 arrays, each of shape (6, 4, 4), in CAM_ORDER.
    """
    if lidar2ego is None:
        lidar2ego = np.eye(4)

    fx = (SRC_W / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)
    fy = fx                                    # CARLA pixels are square
    sx, sy = OUT_W / SRC_W, OUT_H / SRC_H
    K = np.eye(4)
    K[0, 0], K[1, 1] = fx * sx, fy * sy
    K[0, 2], K[1, 2] = (SRC_W / 2.0) * sx, (SRC_H / 2.0) * sy

    out = {k: [] for k in ("camera2lidar", "lidar2camera", "lidar2image",
                           "camera_intrinsics", "camera2ego")}
    for name in CAM_ORDER:
        x, y, z, yaw = CAM_RIG[name]
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = _rot_z(-yaw) @ _R_CAM_TO_LIDAR   # yaw flips sign in a right-handed frame
        cam2lidar[:3, 3] = (x, -y, z)                        # y flips too
        lidar2cam = np.linalg.inv(cam2lidar)
        out["camera2lidar"].append(cam2lidar)
        out["lidar2camera"].append(lidar2cam)
        out["camera_intrinsics"].append(K.copy())
        out["lidar2image"].append(K @ lidar2cam)
        out["camera2ego"].append(lidar2ego @ cam2lidar)

    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}

def lidar2ego_from_offset(lx: float, ly_carla: float, lz: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (lx, -ly_carla, lz)
    return m

def lidar_offset(lidar_transform) -> np.ndarray:
    if lidar_transform is None:
        return np.zeros(3)
    if hasattr(lidar_transform, "location"):
        l = lidar_transform.location
        return np.array([l.x, l.y, l.z], dtype=np.float64)
    return np.asarray(lidar_transform, dtype=np.float64)[:3, 3].copy()

def points_in_rotated_box(points_xyz: np.ndarray, center_x: float, center_y: float, center_z: float,
                           length: float, width: float, height: float, yaw_deg: float, margin: float = 0.1) -> np.ndarray:
    boxes = np.array([[center_x, center_y, center_z, length + (margin * 2), width + (margin * 2),
                       height + (margin * 2), np.deg2rad(yaw_deg)]])
    points_tensor = torch.from_numpy(points_xyz).float()
    boxes_tensor = torch.from_numpy(boxes).float()
    box_point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(boxes_tensor, points_tensor)
    return box_point_masks[0].numpy() > 0

