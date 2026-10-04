from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import open3d as o3d
from enum import IntEnum

class LightState(IntEnum):
    STOP = 0
    NONE = 1
    GO = 2

@dataclass
class Keyframe:
    id: int
    pcd: o3d.geometry.PointCloud          
    world_position_hint: np.ndarray       

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

@dataclass
class SlamOutput:
    """What the EKF receives from the SLAM worker. Everything is time-stamped by frame_id."""
    frame_id: int                     # id of the lidar frame the pose belongs to (EKF step)
    p: np.ndarray                     # (3,) position of the *vehicle* in the SLAM world frame
    q: np.ndarray                     # (4,) [w, x, y, z]
    trans_shape: np.ndarray = field(default_factory=lambda: np.eye(3))  # >=1 inflation shape (vehicle frame)
    map_epoch: int = 0                # increments on every loop-closure optimisation (frame jump)
    fitness: float = 0.0
    inlier_rmse: float = 0.0

@dataclass
class StaticVehicleRef:
    env_id: int
    tag: str
    center_world: np.ndarray   
    size: np.ndarray           
    yaw_world: float           

@dataclass
class Box:
    translation: np.ndarray          
    size: np.ndarray                 
    yaw: float                       
    velocity: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan]))
    tag: str = ""            
    frame_id: str = ""          

    @classmethod
    def from_geometry(cls, x: float, y: float, z: float, length: float, width: float,
                       height: float, yaw: float, vx: float = float("nan"),
                       vy: float = float("nan"), tag: str = "", frame_id: str = "") -> "Box":
        return cls(np.array([x, y, z], dtype=float), np.array([length, width, height], dtype=float),
                    float(yaw), np.array([vx, vy], dtype=float), tag, frame_id)
 
    @classmethod
    def from_trackedbb(cls, tb, frame_id: str = "") -> "Box":
        return cls.from_geometry(tb.px, tb.py, tb.z, tb.len, tb.wid, tb.height, tb.yaw,
                                  vx=tb.vx, vy=tb.vy, frame_id=frame_id)
 
@dataclass
class trackedbb:
    px: float
    py: float
    vx: float
    vy: float
    speed: float
    acceleration: float
    ax: float
    ay: float
    yaw: float
    yaw_rate: float
    len: float
    wid: float
    moveable: bool
    slip_angle_deg: float
    z: float = 0.0
    height: float = 1.5
    pos_history: list[tuple[float, float, float]] = field(default_factory=list)
    score:float = 0.0

@dataclass
class bev_Det:
    center_x: float
    center_y: float
    length: float
    width: float
    yaw: float
    step: int
    center_z: float = 0.0
    height: float = 1.5
    radar_points: list = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0
    score: float = 0.0

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

    forward_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    up_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))

    road_id: int = -1
    lane_id: int = -1
    cross_track_error_m: float = 0      
    heading_error_deg: float = 0        
    speed_limit_ms: float = 0           
    dist_nxt_junc: float = float('inf') 
    light: LightState = LightState.NONE

    detected_objects: list[trackedbb] = field(default_factory=list)
    ttcs: list[float] = field(default_factory=list)
    min_ttc: float = float('inf')

