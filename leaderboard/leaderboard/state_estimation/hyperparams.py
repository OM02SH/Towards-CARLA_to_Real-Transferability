import os
import torch
import numpy as np
import math
import carla

# --- Planar-motion constraint (vz_body ~ 0) and wheel gating
USE_PLANAR_CONSTRAINT = True
SIGMA_WHEEL_VERT = 0.15            # m/s; suspension motion is real, don't make this too tight
WHEEL_MAX_R_SCALE = 5.0            # longitudinal/vertical rows are never down-weighted more than this (was 50)
 
# --- IMU cross-check against wheel/planar kinematics (clamp + process-noise inflation)
IMU_XCHECK = True
IMU_XCHECK_LONG = 5.0              # m/s^2 allowed deviation of IMU-implied dvx/dt from d(wheel speed)/dt
IMU_XCHECK_LAT = 4.0               # m/s^2 allowed |dvy/dt|
IMU_XCHECK_VERT = 3.0              # m/s^2 allowed |dvz/dt|
IMU_Q_INFLATE_MAX = 25.0
 
# --- Published acceleration
ACCEL_OUT_MODE = "kinematic"       # "kinematic": dv_body/dt + w x v (x, y);  "imu": legacy LPF'd IMU
ACCEL_KIN_TAU = 0.10               # s, smoothing of the differentiated velocity
ACCEL_OUT_LPF_TAU = 0.05           # s, now only used for the vertical axis (was 0.15 for everything)
 
# --- Start-up
USE_STARTUP_ALIGNMENT = True       # hold + stationary alignment; check `ekf.initialized` in evaluation
INIT_SKIP_FRAMES = 5               # spawn/settle transient samples ignored
INIT_SETTLE_FRAMES = 10            # stationary samples needed for alignment
INIT_MAX_FRAMES = 60               # give up waiting after this many frames
INIT_ALIGN_TILT = True
INIT_ESTIMATE_GYRO_BIAS = True
SIGMA_GYRO_BIAS_INIT = math.radians(0.2)
 
# --- Bias handling
ESTIMATE_GYRO_BIAS = True
ESTIMATE_ACCEL_BIAS = True         # set False if the CARLA IMU has no accelerometer bias configured
SIGMA_GYRO_BIAS_P0 = math.radians(2.0)
SIGMA_ACCEL_BIAS_P0 = 0.1          # was 0.4
 
# --- SLAM fusion
SLAM_FUSION_MODE = "relative_clone"   # needs SLAM_SYNCHRONOUS = True; otherwise falls back to "relative"
SLAM_ABS_TILT_Z = False               # absolute SLAM z/tilt follows SLAM drift -> hurt position in my tests
SLAM_REL_SCALE_FLOOR = 0.01           # fraction of the displacement (scan-matching scale error)
SLAM_REL_ROT_FLOOR_RAD = math.radians(0.1)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

GNSS_POS_CARLA = (0.0, 0.0, 0.0)
LIDAR_POS_CARLA = (0.0, 0.0, 0.0)
LIDAR_RPY_DEG_CARLA = (0.0, 0.0, 0.0)     
 
# ============================================================================
# EGO STATE
# ============================================================================
G = 9.81
EARTH_RADIUS_EQUA = 6378137.0
 
SIGMA_GNSS_XY = 1
SIGMA_GNSS_Z = 0.45
SIGMA_COMPASS = math.radians(5.0)
SIGMA_WHEEL_LONG = 0.08          
SIGMA_WHEEL_LAT = 0.10   
SIGMA_GRAVITY_DIRECTION = 0.1
SIGMA_GNSS_VEL = 1.5
 
SIGMA_GYRO = math.radians(0.5)
SIGMA_ACCEL = 0.35
SIGMA_GYRO_BIAS_RW = math.radians(0.01)
SIGMA_ACCEL_BIAS_RW = 0.01
SIGMA_WHEEL_SCALE_RW = 1e-3      
 
SIGMA_WHEEL_SCALE_P0 = 0.05
 
GRAVITY_NORM_TOL = 0.5
GRAVITY_UPDATE_MAX_SPEED = 3
GRAVITY_DYNAMIC_SCALE = 4.0
 
ACCEL_CLIP = 40.0                
GYRO_CLIP_RAD = math.radians(100.0)
ACCEL_SPIKE_THRESH = 8.0         
GYRO_SPIKE_THRESH = 1.0          
 
USE_ZUPT = False
ZUPT_WINDOW = 10                 
ZUPT_ACCEL_VAR = 0.05            
ZUPT_GYRO_MAX = 0.03             
ZUPT_WHEEL_MAX = 0.05            
ZUPT_FILTER_SPEED_MAX = 0.3      
SIGMA_ZUPT_VEL = 0.02            
SIGMA_ZUPT_GYRO = math.radians(1.0)
 
SPEEDO_NOISE_STD = 0.05
SPEEDO_QUANT = 0.03
SPEEDO_SCALE_ERR = 0.01          
SPEEDO_DROPOUT = 0.01
 
USE_GNSS_VELOCITY = False
MAX_HISTORY = 20
MIN_HISTORY = 5
 
CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47, 5: 20.52, 6: 22.46}
MAX_R_SCALE = 50.0
GATE_MAX_CONSEC_REJECT = 10      
HISTORY_LEN = 600                
 
ACCEL_OUT_LPF_TAU = 0.15         
OUT_YAW_RATE_SIGN = -1.0         
OUT_Z_SIGN = +1.0                
 
# ============================================================================
# SLAM FUSION
# ============================================================================
SLAM_FUSION_MODE = "relative"    
SLAM_SYNCHRONOUS = True          
 
SLAM_REL_MIN_DT = 0.30           
SLAM_REL_MAX_DT = 1.50           
SIGMA_SLAM_REL_POS = 0.01        
SIGMA_SLAM_REL_ROT = math.radians(0.1)
SLAM_REL_VEL_FLOOR = 0.05        
SLAM_REL_GYRO_FLOOR = 0.01       
SLAM_INFO_COND_CAP = 100.0       

SLAM_ABS_TILT_Z = True
SIGMA_SLAM_TILT = math.radians(1.0)
SIGMA_SLAM_Z = 0.30
 
SIGMA_SLAM_POS = 0.30            
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
 
SUBMAP_KEYFRAMES = 3
USE_ROBUST_KERNEL = True
ICP_ROBUST_K = 0.3
USE_GICP = False
DYNAMIC_BOX_MARGIN = 0.3
DYNAMIC_MIN_SPEED = 0.3
 
 
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
OPENPCDET_SCORE_THRESH = 0.2
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

#################### UKF ###################

POSITION_STD = 1
YAW_STD = math.radians(10.0)
ACCEL_STD = 4.0
YAW_RATE_STD = math.radians(20.0)
L_STD = 0.1
W_STD = 0.1
Z_STD = 0.2
H_STD = 0.2
PROCESS_JERK_STD = 5
PROCESS_JERK_LAT_SCALE = 1.0
PROCESS_YAW_ACCEL_STD = math.radians(20.0)
PROCESS_L_STD = 0.05
PROCESS_W_STD = 0.05
PROCESS_Z_STD = 0.05
PROCESS_H_STD = 0.05
LIDAR_YAW_STD = math.radians(5.0)
LIDAR_L_STD = 0.25
LIDAR_W_STD = 0.25
LIDAR_Z_STD = 0.15
LIDAR_H_STD = 0.15
RADAR_VELOCITY_STD = 0.5
YAW_ALIGN_MIN_SPEED = 2.0
RADAR_GATE_PROB = 0.95
DET_VELOCITY_STD = 1.0

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
