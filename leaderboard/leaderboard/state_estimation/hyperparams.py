import os
import torch
import numpy as np
import math
import carla

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

################ Ego state ##################
G = 9.81

R_GNSS = np.diag([1.5**2, 1.5**2, 1.0**2])
SIGMA_COMPASS = math.radians(5.0)
R_COMPASS = np.array([[SIGMA_COMPASS**2]])
R_WHEEL = np.diag([0.2**2, 0.10**2])

SIGMA_GRAVITY_DIRECTION = 0.15
R_GRAVITY = np.eye(3) * SIGMA_GRAVITY_DIRECTION**2

SIGMA_GYRO = math.radians(0.5)       
SIGMA_ACCEL = 0.4 
SIGMA_GYRO_BIAS_RW = math.radians(0.01)
SIGMA_ACCEL_BIAS_RW = 0.01
SIGMA_WHEEL_BIAS_RW = 0.03           

GRAVITY_NORM_TOL = 0.5             
GRAVITY_UPDATE_MAX_SPEED = 3       
GRAVITY_DYNAMIC_SCALE = 4.0

MAX_HISTORY = 20
MIN_HISTORY = 5
R_GNSS_VEL = np.diag([1.5**2, 1.5**2, 1.5**2])

SIGMA_SLAM_POS = 0.04
SIGMA_SLAM_ATT = math.radians(1.0)

R_SLAM_POS = np.eye(3) * SIGMA_SLAM_POS**2
R_SLAM_ATT = np.eye(3) * SIGMA_SLAM_ATT**2

CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27,}
MAX_R_SCALE = 50.0

EARTH_RADIUS_EQUA = 6378137.0

##################### SLAM #####################
SELF_HIT_RANGE = 0.3
MAX_SENSOR_RANGE = 80.0

TRACK_FITNESS_MIN = 0.35
TRACK_RMSE_MAX = 0.4
TRACK_CORR_DIST = 1.0

EDGE_CORR_DIST_COARSE = 3.0
EDGE_CORR_DIST_FINE = 0.5

KEYFRAME_TRANSLATION_M = 3.0
KEYFRAME_ROTATION_DEG = 8.0

LOOP_SEARCH_RADIUS_M = 15.0
LOOP_MIN_KEYFRAME_GAP = 8          
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
YOLO_MODEL_PATH = os.path.join(_THIS_DIR, "best.pt")

############# Ego state test ###############
POSITION_FIELDS = ("x", "y", "z")
RATE_FIELDS = ("roll_rate", "pitch_rate", "yaw_rate")
VELOCITY_FIELDS = ("vx_body", "vy_body", "vz_body", "speed", "vx_world", "vy_world", "vz_world")
ACCEL_FIELDS = ("ax_body", "ay_body", "az_body", "long_accel", "lat_accel", "vert_accel")
ANGLE_FIELDS = ("roll_rad", "pitch_rad", "yaw_rad")
LINEAR_FIELDS = POSITION_FIELDS + RATE_FIELDS + VELOCITY_FIELDS + ACCEL_FIELDS

STAT_KEYS = ("bias", "std", "mae", "median_abs", "rmse", "p95_abs", "max_abs")

DEFAULT_RPE_DELTA_FRAMES = 20
MOVING_SPEED_THRESHOLD = 1.0       
STATIONARY_SPEED_THRESHOLD = 0.1   
MIN_RPE_SEGMENT = 1.0              

AGGREGATE_ROWS = (
    ("position_2d", "Position 2D (ATE)", "m"),
    ("position_3d", "Position 3D (ATE)", "m"),
    ("position_longitudinal", "Position longitudinal", "m"),
    ("position_lateral", "Position lateral", "m"),
    ("rpe_translation", "RPE translation", "m"),
    ("rpe_translation_percent", "RPE translation", "%"),
    ("rpe_yaw", "RPE yaw", "deg"),
    ("attitude_3d", "Attitude 3D", "deg"),
    ("tilt", "Tilt (roll+pitch)", "deg"),
    ("velocity_world_vec", "Velocity vec (world)", "m/s"),
    ("velocity_body_vec", "Velocity vec (body)", "m/s"),
    ("speed_relative_moving", "Speed rel. (moving)", "%"),
    ("speed_stationary", "Speed (stationary)", "m/s"),
    ("course_angle", "Course angle (moving)", "deg"),
    ("angular_rate_vec", "Angular rate vec", "rad/s"),
    ("accel_body_vec", "Accel vec (body)", "m/s^2"),
)

########### BBs test ###############

VEHICLE_LABELS = [
            getattr(carla.CityObjectLabel, name)
            for name in ("Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Train")
            if hasattr(carla.CityObjectLabel, name)
        ]

DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
TP_DISTANCE_THRESHOLD = 2.0

STATIC_VEHICLE_SUBTAGS = {14: "car", 15: "truck", 16: "bus", 18: "motorcycle", 19: "bicycle"}
