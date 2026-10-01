import os
import torch
import numpy as np
import math

################ Ego state ##################
G = 9.81

R_GNSS = np.diag([2.5**2, 2.5**2, 1.0**2])
SIGMA_COMPASS = math.radians(5.0)
R_COMPASS = np.array([[SIGMA_COMPASS**2]])
R_WHEEL = np.diag([0.25**2, 0.10**2])

SIGMA_GRAVITY_DIRECTION = 0.2
R_GRAVITY = np.eye(3) * SIGMA_GRAVITY_DIRECTION**2

SIGMA_GYRO = math.radians(0.5)       # rad/s/sqrt(Hz)
SIGMA_ACCEL = 0.7                    # m/s^2/sqrt(Hz)
SIGMA_GYRO_BIAS_RW = math.radians(0.01)
SIGMA_ACCEL_BIAS_RW = 0.01
SIGMA_WHEEL_BIAS_RW = 0.03           # m/s/sqrt(Hz)

# Gravity update gating.
GRAVITY_NORM_TOL = 0.5               # m/s^2
GRAVITY_UPDATE_MAX_SPEED = 3       # m/s
GRAVITY_DYNAMIC_SCALE = 4.0

MAX_HISTORY = 20
MIN_HISTORY = 5
R_GNSS_VEL = np.diag([1.5**2, 1.5**2, 1.5**2])

# SLAM typically has centimeter-level precision and sub-degree angular precision.
SIGMA_SLAM_POS = 0.05  # m
SIGMA_SLAM_ATT = math.radians(1.0)

R_SLAM_POS = np.eye(3) * SIGMA_SLAM_POS**2
R_SLAM_ATT = np.eye(3) * SIGMA_SLAM_ATT**2

# Innovation gates.
CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27,}
MAX_R_SCALE = 50.0

##################### SLAM #####################
SELF_HIT_RANGE = 0.3
MAX_SENSOR_RANGE = 80.0

TRACK_FITNESS_MIN = 0.35
TRACK_RMSE_MAX = 0.4
TRACK_CORR_DIST = 1.0

EDGE_CORR_DIST_COARSE = 3.0
EDGE_CORR_DIST_FINE = 0.5

# Keyframe insertion thresholds.
KEYFRAME_TRANSLATION_M = 3.0
KEYFRAME_ROTATION_DEG = 8.0

LOOP_SEARCH_RADIUS_M = 15.0
LOOP_MIN_KEYFRAME_GAP = 8          # ignore recent keyframes as trivial "loop closures"
LOOP_FITNESS_MIN = 0.45
LOOP_RMSE_MAX = 0.3

GLOBAL_OPT_EDGE_PRUNE_THRESHOLD = 0.25


################ Bounding Boxes ################
GROUND_CLEARANCE = 0.3
MAX_LOCKBACK = 50.0
MAX_LOOKFORWARD = 50.0
MAX_LOOKSIDEWAYS = 50.0
MAX_DISTANCE_BETWEEN_LIDAR_POINTS = 0.4
MIN_NUMBER_LIDAR_HITS = 10
MIN_VALID_RANGE_M=0.5
DEFAULT_EGO_HALF_HEIGHT = 0.9
ROAD_EXIT_LOOKAHEAD_S = 6.0
ROAD_EXIT_STEP_S = 0.05

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OPENPCDET_CFG_FILE = os.path.join(_THIS_DIR, "bevfusion_carlaocc.yaml") 
OPENPCDET_CKPT_FILE = os.path.join(_THIS_DIR, "bevfusion.pth") 
OPENPCDET_TOOLS_DIR = os.path.join(_THIS_DIR, "../../../pcdet") 

OPENPCDET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENPCDET_DYNAMIC_CLASSES = {"car", "truck", "pedestrian"}
OPENPCDET_STATIC_CLASSES = {"barrier", "traffic_cone"}
OPENPCDET_SCORE_THRESH = 0.1
OPENPCDET_BOX_MARGIN_M = 0.3     
cam_intrinsic = np.array([
    [552.554261, 0.0, 704.0, 0.0],
    [0.0, 552.554261, 188.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

intrinsics = np.stack([cam_intrinsic] * 6, axis=0)

extrinsics = np.array([
    # cam_00
    [[1.0, 0.0, 0.0, 0.8], [0.0, 1.0, 0.0, 0.3], [0.0, 0.0, 1.0, -0.18], [0.0, 0.0, 0.0, 1.0]],
    # cam_01
    [[1.0, 0.0, 0.0, 0.8], [0.0, 1.0, 0.0, -0.3], [0.0, 0.0, 1.0, -0.18], [0.0, 0.0, 0.0, 1.0]],
    # cam_02
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.46], [0.0, 0.0, 1.0, -0.18], [0.0, 0.0, 0.0, 1.0]],
    # cam_03
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -0.46], [0.0, 0.0, 1.0, -0.18], [0.0, 0.0, 0.0, 1.0]],
    # cam_04
    [[-1.0, 0.0, 0.0, -0.8], [0.0, -1.0, 0.0, 0.3], [0.0, 0.0, 1.0, -0.18], [0.0, 0.0, 0.0, 1.0]],
    # cam_05
    [[-1.0, 0.0, 0.0, -0.8], [0.0, -1.0, 0.0, -0.3], [0.0, 0.0, 1.0, -0.18], [0.0, 0.0, 0.0, 1.0]]
], dtype=np.float32)

############## Traffic Light ###############
STOP_CLASS_ID = 0
GO_CLASS_ID = 1
EXPECTED_ASPECT_RATIO = 0.273
ASPECT_RATIO_TOLERANCE = 0.20
HOLD_MAX_CALLS = 10

############# Ego state test ###############

LINEAR_FIELDS = ("x", "y", "z",
                 "roll_rate", "pitch_rate", "yaw_rate",
                 "vx_body", "vy_body", "vz_body", "speed",
                 "vx_world", "vy_world", "vz_world",
                 "ax_body", "ay_body", "az_body",
                 "long_accel", "lat_accel", "vert_accel")
ANGLE_FIELDS = ("roll_rad", "pitch_rad", "yaw_rad")
