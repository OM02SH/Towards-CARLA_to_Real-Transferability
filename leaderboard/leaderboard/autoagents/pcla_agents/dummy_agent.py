"""
Dummy PCLA-style waypoint-following agent.

Infinite lane-following agent: Follows the centerline of its current lane 
indefinitely using CARLA's map topology without stopping when reaching the 
end of a global plan.

Entry point for the CARLA Leaderboard evaluator:
    export TEAM_AGENT=dummy_pcla_agent.py
    export TEAM_CONFIG=  (unused, no config needed)
"""

import math

import numpy as np
import carla

from leaderboard_codes.autonomous_agent2 import AutonomousAgent, Track
from leaderboard_codes.carla_data_provider import CarlaDataProvider

from pcla_agents.simlingo.nav_planner import RoutePlanner, PIDController, _get_latlon_ref


def get_entry_point():
    return "DummyWaypointAgent"


def _wrap_angle(angle):
    """Wrap an angle (radians) to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class DummyWaypointAgent(AutonomousAgent):
    """
    Follows the current lane's centerline indefinitely at a constant 
    target speed using basic PID steering on heading error.
    """

    def setup(self, path_to_conf_file, route_index):
        self.track = Track.SENSORS
        self.step = -1
        self._target_speed = 5.0  # m/s, constant dummy cruising speed
        self._lookahead_distance = 7.5  # meters ahead on the centerline

        world_map = CarlaDataProvider.get_map()
        lat_ref, lon_ref = _get_latlon_ref(world_map)

        self._world_map = world_map
        self._route_planner = RoutePlanner(
            min_distance=7.5, max_distance=50.0, lat_ref=lat_ref, lon_ref=lon_ref
        )
        self._steer_pid = PIDController(k_p=1.0, k_i=0.0, k_d=0.1, n=20)

    def sensors(self):
        return [
            {
                "type": "sensor.other.gnss",
                "x": 0.0, "y": 0.0, "z": 2.0,
                "id": "GPS",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0, "y": 0.0, "z": 2.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "id": "IMU",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "SPEED",
            },
        ]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def run_step(self, input_data, timestamp=None, vehicle=None):
        self.step += 1

        _, gps_data = input_data["GPS"]
        _, imu_data = input_data["IMU"]
        _, speed_data = input_data["SPEED"]

        gps_carla = self._route_planner.convert_gps_to_carla(gps_data)
        compass = imu_data[-1]  # radians, CARLA convention: 0 = North
        speed = speed_data["speed"] if isinstance(speed_data, dict) else speed_data[0]

        # 1. Query map for the current lane centerline waypoint from GPS position
        ego_loc = carla.Location(x=float(gps_carla[0]), y=float(gps_carla[1]), z=float(gps_carla[2]))
        current_wp = self._world_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # 2. Advance along the current lane's centerline topology
        next_wps = current_wp.next(self._lookahead_distance)
        target_wp = next_wps[0] if next_wps else current_wp

        target_loc = target_wp.transform.location
        target_pos = np.array([target_loc.x, target_loc.y, target_loc.z])
        route = [(target_pos, None)]

        # 3. Compute Steering Control
        delta = target_pos[:2] - gps_carla[:2]
        target_heading = math.atan2(delta[1], delta[0])

        ego_heading = (math.pi / 2.0) - compass
        heading_error = _wrap_angle(target_heading - ego_heading) / math.pi

        control = carla.VehicleControl()
        control.steer = float(np.clip(self._steer_pid.step(heading_error), -1.0, 1.0))

        # 4. Compute Longitudinal Control
        speed_error = self._target_speed - speed
        if speed_error > 0:
            control.throttle = float(np.clip(speed_error, 0.0, 0.75))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(np.clip(-speed_error, 0.0, 1.0))

        return control, route, "holder"

    def destroy(self):
        pass