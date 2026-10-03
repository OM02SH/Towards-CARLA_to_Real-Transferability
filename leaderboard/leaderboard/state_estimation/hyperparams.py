import os
import torch
import numpy as np
import math
import carla

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# SENSOR MOUNTS (CARLA vehicle-relative coordinates, exactly as in your
# sensors() dict: x forward, y RIGHT, z up, angles in degrees).
# !!! FILL THESE IN from your sensor configuration. (0,0,0) = no lever arm. !!!
# ============================================================================
GNSS_POS_CARLA = (0.0, 0.0, 0.0)
LIDAR_POS_CARLA = (0.0, 0.0, 0.0)
LIDAR_RPY_DEG_CARLA = (0.0, 0.0, 0.0)     # roll, pitch, yaw
 
# ============================================================================
# EGO STATE
# ============================================================================
G = 9.81
EARTH_RADIUS_EQUA = 6378137.0
 
# ---- measurement noise (1-sigma). Ideally = the noise you configured in CARLA.
SIGMA_GNSS_XY = 1.0
SIGMA_GNSS_Z = 0.75
SIGMA_COMPASS = math.radians(5.0)
SIGMA_WHEEL_LONG = 0.08          # was 0.2; sim speedometer noise is 0.05 + quantisation
SIGMA_WHEEL_LAT = 0.10           # non-holonomic constraint
SIGMA_GRAVITY_DIRECTION = 0.1
SIGMA_GNSS_VEL = 1.5
 
# ---- process noise
SIGMA_GYRO = math.radians(0.5)
SIGMA_ACCEL = 0.35
SIGMA_GYRO_BIAS_RW = math.radians(0.01)
SIGMA_ACCEL_BIAS_RW = 0.01
SIGMA_WHEEL_SCALE_RW = 1e-3      # wheel speed *scale factor* random walk (1/sqrt(s))
 
# ---- initial covariance for the wheel scale-factor state (dimensionless)
SIGMA_WHEEL_SCALE_P0 = 0.05
 
# ---- gravity-direction update
GRAVITY_NORM_TOL = 0.5
GRAVITY_UPDATE_MAX_SPEED = 3
GRAVITY_DYNAMIC_SCALE = 4.0
 
# ---- IMU pre-processing
ACCEL_CLIP = 40.0                # safety clip only (was +-10, which clipped gravity+noise on z)
GYRO_CLIP_RAD = math.radians(100.0)
ACCEL_SPIKE_THRESH = 8.0         # m/s^2   Hampel-style spike rejection (median of 3)
GYRO_SPIKE_THRESH = 1.0          # rad/s
 
# ---- ZUPT (zero-velocity update) / stationary detection
USE_ZUPT = True
ZUPT_WINDOW = 10                 # samples
ZUPT_ACCEL_VAR = 0.05            # sum of per-axis accel variance over window
ZUPT_GYRO_MAX = 0.03             # rad/s, |mean(gyro) - bias|
ZUPT_WHEEL_MAX = 0.05            # m/s, |wheel speed| below this counts as standing
ZUPT_FILTER_SPEED_MAX = 0.3      # used only if wheel speed dropped out
SIGMA_ZUPT_VEL = 0.02            # m/s
SIGMA_ZUPT_GYRO = math.radians(1.0)
 
# ---- simulated speedometer (get_simulated_speedometer)
SPEEDO_NOISE_STD = 0.05
SPEEDO_QUANT = 0.03
SPEEDO_SCALE_ERR = 0.01          # multiplicative error (real encoders: scale, not offset)
SPEEDO_DROPOUT = 0.01
 
# ---- GNSS-regression velocity pseudo-measurement (double-counts GNSS; OFF by default)
USE_GNSS_VELOCITY = False
MAX_HISTORY = 20
MIN_HISTORY = 5
 
# ---- robust handling
CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47, 5: 20.52, 6: 22.46}
MAX_R_SCALE = 50.0
GATE_MAX_CONSEC_REJECT = 10      # hard-gated sensor is force-accepted (inflated R) after N rejections
HISTORY_LEN = 600                # EKF steps kept for delayed / windowed SLAM fusion
 
# ---- output (extract_state_from_filter)
ACCEL_OUT_LPF_TAU = 0.15         # s, first-order low-pass on the reported accelerations
# Sign conventions of the *reported* fields. Verify each with a regression of
# estimate vs GT (slope must be +1) and flip here if needed:
OUT_YAW_RATE_SIGN = -1.0         # yaw_rad is reported as -yaw_ekf, so yaw_rate must flip too
OUT_Z_SIGN = +1.0                # z is not mirrored between CARLA (LH) and the EKF (RH)
 
# ============================================================================
# SLAM FUSION
# ============================================================================
SLAM_FUSION_MODE = "relative"    # "relative" (drift-free, recommended) | "absolute"
SLAM_SYNCHRONOUS = True          # True: deterministic (no thread). False: async worker
 
# ---- relative mode: non-overlapping windows of consecutive SLAM poses
SLAM_REL_MIN_DT = 0.30           # s, minimum baseline for one measurement
SLAM_REL_MAX_DT = 1.50           # s, longer gaps -> just restart the window
SIGMA_SLAM_REL_POS = 0.01        # m, per-scan relative translation noise
SIGMA_SLAM_REL_ROT = math.radians(0.1)
SLAM_REL_VEL_FLOOR = 0.05        # m/s   model-mismatch floor
SLAM_REL_GYRO_FLOOR = 0.01       # rad/s
SLAM_INFO_COND_CAP = 100.0       # max inflation of weakly constrained ICP directions
# Relative mode has no absolute roll/pitch/z while moving. Scan-to-map ICP constrains them well
# (ground plane) and they drift slowly, so fuse them loosely as absolute (delay-compensated) measurements.
SLAM_ABS_TILT_Z = True
SIGMA_SLAM_TILT = math.radians(1.0)
SIGMA_SLAM_Z = 0.30
 
# ---- absolute mode (only used if SLAM_FUSION_MODE == "absolute")
SIGMA_SLAM_POS = 0.30            # was 0.03: SLAM error is drift-like, not white
SIGMA_SLAM_ATT = math.radians(2.0)
 
# ============================================================================
# SLAM FRONT END
# ============================================================================
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
 
SUBMAP_KEYFRAMES = 3             # scan-to-submap (last N keyframes) instead of last keyframe only
USE_ROBUST_KERNEL = True         # Tukey loss in point-to-plane ICP
ICP_ROBUST_K = 0.3               # m
USE_GICP = False                 # needs Open3D >= 0.17; falls back automatically
DYNAMIC_BOX_MARGIN = 0.3         # m, inflation of dynamic-object boxes
DYNAMIC_MIN_SPEED = 0.3          # m/s, CARLA helper: actors slower than this are kept in the map
 
 
# ============================================================================
# Derived matrices
# ============================================================================
def rebuild():
    g = globals()
    g["R_GNSS"] = np.diag([SIGMA_GNSS_XY ** 2, SIGMA_GNSS_XY ** 2, SIGMA_GNSS_Z ** 2])
    g["R_COMPASS"] = np.array([[SIGMA_COMPASS ** 2]])
    g["R_WHEEL"] = np.diag([SIGMA_WHEEL_LONG ** 2, SIGMA_WHEEL_LAT ** 2])
    g["R_GRAVITY"] = np.eye(3) * SIGMA_GRAVITY_DIRECTION ** 2
    g["R_GNSS_VEL"] = np.eye(3) * SIGMA_GNSS_VEL ** 2
    g["R_ZUPT_VEL"] = np.eye(3) * SIGMA_ZUPT_VEL ** 2
    g["R_ZUPT_GYRO"] = np.eye(3) * SIGMA_ZUPT_GYRO ** 2
    g["R_SLAM_POS"] = np.eye(3) * SIGMA_SLAM_POS ** 2
    g["R_SLAM_ATT"] = np.eye(3) * SIGMA_SLAM_ATT ** 2
 
 
rebuild()

################ Bounding Boxes ################


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

R_CAM_TO_LIDAR = np.array([[0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]], dtype=np.float64)

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
OPENPCDET_DYNAMIC_CLASSES = {"car", "truck", "pedestrian", "bus", 'motorcycle','bicycle', 'train'}
OPENPCDET_STATIC_CLASSES = {"wall", "fence", "guard_rail"}
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
