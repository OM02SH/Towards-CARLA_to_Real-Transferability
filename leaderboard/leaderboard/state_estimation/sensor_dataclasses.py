from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter
import open3d as o3d
from enum import IntEnum


@dataclass
class Keyframe:
    id: int
    pcd: o3d.geometry.PointCloud          # local sensor-frame cloud, WITH normals
    world_position_hint: np.ndarray       # last-known world position, for loop search only


@dataclass
class SlamUpdateResult:
    success: bool
    p_slam: Optional[np.ndarray] = None
    q_slam: Optional[np.ndarray] = None
    fitness: float = 0.0
    inlier_rmse: float = 0.0
    info_matrix: Optional[np.ndarray] = None
    loop_closed: bool = False
    num_keyframes: int = 0
    reason: str = ""

class LightState(IntEnum):
    GO = 1
    STOP = 2
    NONE = 3   # no result yet, or no light detected in the latest processed frame

@dataclass
class trackedbb:
    px : float
    py : float
    vx : float
    vy : float
    speed : float
    acceleration : float
    ax : float
    ay : float
    yaw : float
    yaw_rate : float
    len : float
    wid : float
    moveable: bool
    slip_angle_deg : float
    pos_history : list[tuple[float,float,float]] = field(default_factory=list)

@dataclass
class BoundingBox2D:
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    yaw: float
    tick_count: int              # tick index this detection was observed at
    radar_point: np.ndarray     # [x, y, v_r_abs, u_x, u_y]

@dataclass
class UKFBoundingBox2D:
    center_x: float
    center_y: float

    length: float
    width: float

    yaw: float
    step: int
    radar_points: list = field(default_factory=list)

@dataclass
class Trackbb2D:
    id: int
    bbs: List[BoundingBox2D] = field(default_factory=list)
    vx: Optional[float] = None
    vy: Optional[float] = None
    missed_ticks: int = 0       
    lastest_tick: int = 0
    heading: Tuple[float,float,bool] = None
    vel_history : List[tuple] = field(default_factory=list)  
    accel : Tuple[float,bool] = None

@dataclass
class TrackBB2D:
    id: int
    bbs: List[UKFBoundingBox2D]
    kf: KalmanFilter
    missed_ticks: int = 0
    vx: float = 0.0
    vy: float = 0.0
    heading: tuple = (0.0, 0.0, False)
    accel: tuple = (0.0, False)
    vel_history: list = field(default_factory=list)

@dataclass
class UKFState:
    x: float = 0                    # Global X position (m)
    y: float = 0                    # Global Y position (m)
    yaw_rad: float = 0              # Heading angle in standard right-handed ENU (rad)
    yaw: float = 0                  # Heading angle converted back to CARLA frame (deg)
    vx_body: float = 0              # Forward longitudinal speed in ego frame (m/s)
    vy_body: float = 0              # Sideways lateral speed in ego frame (m/s)
    yaw_rate: float = 0             # Estimated true Yaw rate (rad/s)
    ax_body: float = 0              # Forward longitudimal acceleration in ego frame (m/s²)
    ay_body: float = 0              # Sideways lateral acceleration in ego frame (m/s²)    

    speed: float = 0                # Scalar speed v = sqrt(vx² + vy²) (m/s)
    sideslip_angle_deg: float = 0   # Beta = arctan2(vy, vx) (deg)
    vx_world: float = 0             # Global map velocity X (m/s)
    vy_world: float = 0             # Global map velocity Y (m/s)

    long_accel: float = 0           # Longitudinal acceleration (m/s²)
    lat_accel: float = 0            # Lateral acceleration (m/s²)
    accel_mag: float = 0            # acceleration magnitude = sqrt(long_accel + lat_accel) (m/s²)
    lat_jerk: float = 0             # Longitudinal Jerk da_x / dt (m/s³)
    long_jerk: float = 0            # Lateral Jerk da_y / dt (m/s³)
    jerk_mag: float = 0             # Jerk magnitude = sqrt(jerk_x² + jerk_y²) (m/s³)

    sigma_pos: float = 0            # Position uncertainty sqrt(σ_x² + σ_y²)
    sigma_speed: float = 0          # Speed uncertainty σ_vx
    sigma_yaw: float = 0            # Heading uncertainty σ_psi (deg)


"""@dataclass
class UKFState:
    x: float                     # Global X position (m)
    y: float                     # Global Y position (m)
    z: float                     # Global Z position (m)

    roll_rad:float               # roll angle (rad)
    pitch_rad:float              # pitch angle (rad)
    yaw_rad: float               # Heading angle in standard right-handed ENU (rad)

    roll:float                   # roll angle (deg)
    pitch:float                  # pitch angle (deg)
    yaw: float                   # Heading angle converted back to CARLA frame (deg)

    roll_rate: float             # Estimated true roll rate (rad/s)
    pitch_rate: float            # Estimated true pitch rate (rad/s)
    yaw_rate: float              # Estimated true Yaw rate (rad/s)

    vx_body: float               # Forward longitudinal speed in ego frame (m/s)
    vy_body: float               # Sideways lateral speed in ego frame (m/s)
    vz_body: float               # Climbing speed in ego frame (m/s)

    ax_body: float               # Forward longitudinal acceleration in ego frame (m/s²)
    ay_body: float               # Sideways lateral acceleration in ego frame (m/s²)
    az_body: float               # Climbing acceleration in ego frame (m/s²)

    speed: float                 # Scalar total speed v = sqrt(vx² + vy² + vz²) (m/s)
    horizontal_speed: float      # Scalar horizontal speed v = sqrt(vx² + vy²) (m/s)
    sideslip_angle_rad: float    # Beta = arctan2(vy, vx) (rad)
    sideslip_angle: float        # Beta = arctan2(vy, vx) (deg)

    vx_world: float              # Global map velocity X (m/s)
    vy_world: float              # Global map velocity Y (m/s)
    vz_world: float              # Global map velocity Z (m/s)

    long_accel: float            # Longitudinal acceleration (m/s²)
    lat_accel: float             # Lateral acceleration (m/s²)
    vert_accel: float            # Vertical acceleration (m/s²)
    
    long_jerk: float             # Longitudinal Jerk da_x / dt (m/s³)
    lat_jerk: float              # Lateral Jerk da_y / dt (m/s³)
    vert_jerk: float             # Vertical Jerk da_z / dt (m/s³)"""

@dataclass
class Ego3DState:
    x: float
    y: float
    z: float

    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    roll: float
    pitch: float
    yaw: float

    roll_rate: float
    pitch_rate: float
    yaw_rate: float

    vx_body: float
    vy_body: float
    vz_body: float

    ax_body: float
    ay_body: float
    az_body: float

    speed: float

    vx_world: float
    vy_world: float
    vz_world: float
    
    long_accel: float
    lat_accel: float 
    vert_accel: float

    bg_yaw: float
    bg_pitch: float
    bg_roll: float

@dataclass
class CalculatedState:
    ego_state : Ego3DState = None    
    g_force_longitudinal: float = 0  
    g_force_lateral: float = 0      

    forward_vector_est: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right_vector_est: np.ndarray = field(default_factory=lambda: np.zeros(3))

    road_id: int = -1
    lane_id: int = -1
    cross_track_error_m: float = 0      # distance from the lane center   
    heading_error_deg: float = 0        # heading error with lane refrence 
    speed_limit_ms: float = 0           # current lane speed limit (privilged)
    dist_nxt_junc: float = float('inf') # distance to next junctior

    detected_objects: list[trackedbb] = field(default_factory=list)
    ttcs: list[float] = field(default_factory=list)
    min_ttc: float = float('inf')

