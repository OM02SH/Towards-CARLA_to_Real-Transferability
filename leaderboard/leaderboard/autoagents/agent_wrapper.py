#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os
import time
import sys
import gc

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.ros_base_agent import ROSBaseAgent

import traceback
import torch
import importlib
from leaderboard.autoagents.pcla_functions import give_path, setup_sensor_attributes, location_to_waypoint, route_maker
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_indexer import RouteIndexer
from srunner.scenariomanager.watchdog import Watchdog


IS_BENCH2DRIVE = os.environ.get('SAVE_PATH', None)
if IS_BENCH2DRIVE:
    MAX_ALLOWED_RADIUS_SENSOR = 100.0  # for visualize
else:
    MAX_ALLOWED_RADIUS_SENSOR = 3.0

QUALIFIER_SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}
SENSORS_LIMITS = {
    'sensor.camera.rgb': 8,
    'sensor.lidar.ray_cast': 2,
    'sensor.other.radar': 4,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}
ALLOWED_SENSORS = SENSORS_LIMITS.keys()


class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)

        
class TickRuntimeError(Exception):
    pass

class AgentWrapperFactory(object):

    @staticmethod
    def get_wrapper(agent):
        if isinstance(agent, ROSBaseAgent):
            return ROSAgentWrapper(agent)
        else:
            return AgentWrapper(agent)


def validate_sensor_configuration(sensors, agent_track, selected_track):
    """
    Ensure that the sensor configuration is valid, in case the challenge mode is used
    Returns true on valid configuration, false otherwise
    """
    if Track(selected_track) != agent_track:
        raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

    sensor_count = {}
    sensor_ids = []

    for sensor in sensors:

        # Check if the is has been already used
        sensor_id = sensor['id']
        if sensor_id in sensor_ids:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
        else:
            sensor_ids.append(sensor_id)

        # Check if the sensor is valid
        if agent_track == Track.SENSORS:
            if sensor['type'].startswith('sensor.opendrive_map'):
                raise SensorConfigurationInvalid("Illegal sensor 'sensor.opendrive_map' used for Track [{}]!".format(agent_track))

        # Check the sensors validity
        if sensor['type'] not in ALLOWED_SENSORS:
            raise SensorConfigurationInvalid("Illegal sensor '{}' used for Track [{}]!".format(sensor['type'], agent_track))

        # Check the extrinsics of the sensor
        if 'x' in sensor and 'y' in sensor and 'z' in sensor:
            if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                raise SensorConfigurationInvalid(
                    "Illegal sensor extrinsics used for sensor '{}'. Max allowed radius is {}m".format(sensor['id'], MAX_ALLOWED_RADIUS_SENSOR))

        # Check the amount of sensors
        if sensor['type'] in sensor_count:
            sensor_count[sensor['type']] += 1
        else:
            sensor_count[sensor['type']] = 1

    if agent_track in (Track.SENSORS_QUALIFIER, Track.MAP_QUALIFIER):
        sensor_limits = QUALIFIER_SENSORS_LIMITS
    else:
        sensor_limits = SENSORS_LIMITS

    for sensor_type, max_instances_allowed in sensor_limits.items():
        if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
            raise SensorConfigurationInvalid(
                "Too many {} used! "
                "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                              max_instances_allowed,
                                                                              sensor_count[sensor_type]))


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """
    _agent = None
    _sensors_list = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent()

    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == 'sensor.camera.rgb':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)
            attributes['rotation_frequency'] = str(10)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(600000)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'  # meters

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            # These are the pseudosensors (not spawned)
            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)

            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        for _ in range(10):
            world.tick()

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []

        # Tick once to destroy the sensors
        CarlaDataProvider.get_world().tick()

class PCLA():
    def __init__(self, agent, vehicle, config, client):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.client = None
        self.world = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.agent_instance = None
        self.config = None
        self._watchdog = None
        self.routePath = None
        # ==========================================================
        # === Latency Injection Configuration & State Initialization ===
        # ==========================================================
        self._inf_latency_enabled = int(os.environ.get("INFERENCE_LATENCY_ENABLE", "0")) != 0
        self._inf_latency_ms = float(os.environ.get("INFERENCE_LATENCY_MS", "0"))
        self._sim_rate = float(os.environ.get("SIM_RATE", "20"))  
        self._warmup_steps = int(os.environ.get("WARMUP_STEPS", "20"))
        self._latency_mode = os.environ.get("INFERENCE_LATENCY_MODE", "fixed").lower()
        
        self.wall_start = time.time()
        self._step = 0
        self._first_inference = True
        self._next_inference_step = -1
        self._next_control = None
        self._current_control = None
        self._last_measured_inference_time = 0.0
        
        # Default control (stationary) during warmup/delay
        self._default_control = carla.VehicleControl()
        self._default_control.steer = 0.0
        self._default_control.throttle = 0.0
        self._default_control.brake = 0.0
        # ==========================================================
        self.set(agent, vehicle, config, client)
    
    def set(self, agent, vehicle, config, client):
        self.client = client
        self.world = client.get_world()
        self.vehicle = vehicle
        self.config = config
        self._watchdog = Watchdog(260) # TODO: Increase timeout if needed for large models
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        # Reset latency state when re-setting an agent
        self.wall_start = time.time()
        self._step = 0
        self._first_inference = True
        self._next_inference_step = -1
        self._next_control = None
        self._current_control = None
        self._last_measured_inference_time = 0.0

        self.setup_agent(agent)
        self.setup_route()
        self.setup_sensors()

    def setup_agent(self, agent):
        GameTime.restart()
        self._watchdog.start()
        self.agentPath, self.configPath = give_path(agent, self.current_dir, self.routePath)
        # Reload the agent freshly each time to avoid stale imports across agents.
        # Use file-based loading to prevent cross-talk between agents sharing module names (e.g., model.py).
        module_name = os.path.basename(self.agentPath).split('.')[0]
        module_dir = os.path.dirname(self.agentPath)
        module_key = f"pcla_dynamic_agent.{module_name}"

        # Prepend the agent directory so its relative imports resolve to local files.
        # Save/restore sys.path to isolate dependencies per agent.
        original_sys_path = list(sys.path)
        if module_dir in sys.path:
            sys.path.remove(module_dir)
        sys.path.insert(0, module_dir)

        # Drop previously loaded local agent modules to avoid cross-agent contamination
        # (e.g., plant2's dataset being reused by plant1).
        module_names_to_clear = {
            module_key,
            'model', 'models', 'dataset', 'lit_module', 'plant_variables',
            'nav_planner', 'config', 'transfuser', 'transfuser_utils',
            'utils', 'util', 'data', 'gaussian_target',
            'planner', 'controller', 'map_agent', 'base_agent',
            'bev_planner', 'waypointer', 'lateral_controller',
            'longitudinal_controller', 'kinematic_bicycle_model'
        }
        module_prefixes_to_clear = (
            'models.',
            'util.',
            'carla_garage.',
            'birds_eye_view.',
        )
        for key in list(sys.modules.keys()):
            if key in module_names_to_clear or key == 'carla_garage' or key.startswith(module_prefixes_to_clear):
                del sys.modules[key]

        try:
            spec = importlib.util.spec_from_file_location(module_key, self.agentPath)
            module_agent = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module_agent
            spec.loader.exec_module(module_agent)
        finally:
            # Restore sys.path
            sys.path = original_sys_path
        
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        self.agent_instance = getattr(module_agent, agent_class_name)(self.configPath)

        self._watchdog.stop()

    def setup_route(self):
        """scenarios = os.path.join(self.current_dir, "leaderboard_codes/no_scenarios.json")
        route_indexer = RouteIndexer(self.routePath, scenarios, 1)
        config = route_indexer.next()"""
        
        gps_route, route = interpolate_trajectory(self.world, self.config.trajectory)

        self.agent_instance.set_global_plan(gps_route, route)

    def setup_sensors(self):
        """Attach sensors defined by the agent to the ego-vehicle."""
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in self.agent_instance.sensors():
            # Pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                sensor = OpenDriveMapReader(self.vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = 1/20
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(self.vehicle, frame_rate)
            else:
                # World sensors (spawned actors)
                bp = bp_library.find(str(sensor_spec['type']))
                bp_setup = setup_sensor_attributes(bp, sensor_spec)
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z'])
                if sensor_spec['type'].startswith('sensor.other.gnss'):
                    sensor_rotation = carla.Rotation()
                else:
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'], roll=sensor_spec['roll'], yaw=sensor_spec['yaw'])

                # Create sensor actor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp_setup, sensor_transform, self.vehicle)
            # Register callback
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.agent_instance.sensor_interface))

        # Ensure sensors are created in the world
        self.world.tick()
        CarlaDataProvider.register_actor(self.vehicle)
    # ==========================================================
    # === Latency Helper Methods ===
    # ==========================================================
    def _get_sensor_data(self, input_data: dict, sensor_name: str):
        return input_data.get(sensor_name, input_data.get(sensor_name.upper(), None))

    def _sensor_data_ready(self) -> bool:
        try:
            input_data = self.agent_instance.sensor_interface.get_data(GameTime.get_frame())
            speed_data = self._get_sensor_data(input_data, 'speed')
            speed_ok = speed_data is not None

            gps_data = self._get_sensor_data(input_data, 'gps')
            gps_ok = False
            if gps_data is not None:
                try:
                    val = gps_data[1]
                    if val is not None:
                        if isinstance(val, dict):
                            gps_ok = ('lat' in val and val['lat'] is not None and 'lon' in val and val['lon'] is not None)
                        else:
                            gps_ok = (hasattr(val, '__len__') and len(val) >= 2 and val[0] is not None and val[1] is not None)
                except Exception:
                    gps_ok = False

            return speed_ok and gps_ok
        except Exception:
            return False

    def _run_agent_with_timing(self):
        inference_start_time = time.time()
        control = self.agent_instance(vehicle=self.vehicle)
        self._last_measured_inference_time = time.time() - inference_start_time
        return control

    def _call_measured_latency(self):
        if self._step <= self._warmup_steps or not self._sensor_data_ready():
            if self._step > self._warmup_steps and not self._sensor_data_ready():
                print(f"[INFERENCE LATENCY] step={self._step} WAITING_FOR_SENSOR_DATA", flush=True)
            return self._default_control

        if self._first_inference:
            self._next_control = self._run_agent_with_timing()
            self._next_inference_step = self._step + int(self._last_measured_inference_time * self._sim_rate)
            self._first_inference = False
            print(f"[INFERENCE LATENCY] step={self._step} FIRST_INFERENCE measured_inference_time={self._last_measured_inference_time:.6f}s "
                  f"next_control_step={self._next_inference_step} (result delayed)", flush=True)
            return self._default_control

        if self._step == self._next_inference_step:
            self._current_control = self._next_control
            self._next_control = self._run_agent_with_timing()
            self._next_inference_step = self._step + int(self._last_measured_inference_time * self._sim_rate)
            print(f"[INFERENCE LATENCY] step={self._step} SWITCH_BUFFER measured_inference_time={self._last_measured_inference_time:.6f}s "
                  f"next_control_step={self._next_inference_step}", flush=True)
            return self._current_control

        if self._current_control is not None:
            return self._current_control
        return self._default_control
    # ==========================================================

    def get_action(self, Qs = None):
        """
        Pass the call to the agent with inference latency scheduling

        Logic:
        1. Extended warm-up phase: wait for sensor data to be ready
        2. First inference: result stored in next_control, current_control still None
        3. Wait for delay_ticks ticks
        4. At next_inference_step: next_control -> current_control, start using it, infer next
        5. Loop: each inference result is used after delay_ticks ticks
        """

        snapshot = self.world.get_snapshot()
        if snapshot:
            timestamp = snapshot.timestamp
            if timestamp:
                GameTime.on_carla_tick(timestamp)
        self._step += 1

        # === Print Tick status ===
        # wallclock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # game_time = GameTime.get_time()
        # print(f"=== [Tick] step={self._step} -- Wallclock = {wallclock} -- System time = {time.time() - self.wall_start:.3f} -- Game time = {game_time:.3f} -- Ratio = {game_time / (time.time() - self.wall_start + 1e-6):.3f}x")

        # If inference latency disabled, call agent directly
        if not self._inf_latency_enabled:
            return self.agent_instance(vehicle = self.vehicle)
        # Measured mode: use the agent's real runtime to schedule control release
        if self._latency_mode == "measured":
            return self._call_measured_latency()
        if self._inf_latency_ms <= 0: 
            return self.agent_instance(vehicle = self.vehicle)

        # Calculate delay ticks
        delay_ticks = int(self._inf_latency_ms * self._sim_rate / 1000.0)

        # === Extended warm-up: wait for sensor data to be ready ===
        if self._step <= self._warmup_steps or not self._sensor_data_ready():
            if self._step > self._warmup_steps and not self._sensor_data_ready():
                print(f"[INFERENCE LATENCY] step={self._step} WAITING_FOR_SENSOR_DATA", flush=True)
            return self._default_control   
        
        # === First inference ===
        if self._first_inference:
            # Store inference result in next_control (not used yet)
            self._next_control = self.agent_instance(vehicle = self.vehicle)
            # Calculate next switch time: next -> current
            self._next_inference_step = self._step + delay_ticks
            self._first_inference = False
            #print(f"[INFERENCE LATENCY] step={self._step} FIRST_INFERENCE delay_ticks={delay_ticks} "
            #      f"next_control_step={self._next_inference_step} (result delayed)", flush=True)
            # current_control still None, return default control
            return self._default_control

        # === At next_inference_step: switch buffer and infer next ===
        if self._step == self._next_inference_step:
            # next -> current, start using previous inference result
            self._current_control = self._next_control
            # Infer next, store in next
            self._next_control = self.agent_instance(vehicle = self.vehicle, Qs = Qs)
            # Calculate next switch time
            self._next_inference_step = self._step + delay_ticks
            #print(f"[INFERENCE LATENCY] step={self._step} SWITCH_BUFFER delay_ticks={delay_ticks} "
            #      f"next_control_step={self._next_inference_step}", flush=True)
            return self._current_control

        # === Otherwise: use current_control ===
        if self._current_control is not None:
            return self._current_control
        else:
            return self._default_control
    
    def cleanup(self):
        """Remove and destroy all actors."""

        if self._watchdog:
            self._watchdog.stop()

        # Cleanup the agent first so it can stop any internal threads and sensors
        try:
            if self.agent_instance is not None:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        # Stop and destroy any remaining sensors
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            if sensor.is_listening():
                sensor.stop()
            sensor.destroy()

        # Destroy the vehicle after sensors are cleaned up
        try:
            if self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.destroy()
        except RuntimeError:
            pass

        self.current_dir = None
        self.client = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.routePath = None
        self.world = None
        
        # Release cached CUDA memory between agents to avoid cross-agent OOMs.
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        CarlaDataProvider.cleanup()


class ROSAgentWrapper(AgentWrapper):

    SENSOR_TYPE_REMAPS = {
        "sensor.opendrive_map": "sensor.pseudo.opendrive_map",
        "sensor.speedometer": "sensor.pseudo.speedometer"
    }

    def __init__(self, agent):
        super(ROSAgentWrapper, self).__init__(agent)

    def _preprocess_sensor_spec(self, sensor_spec):
        type_, id_, sensor_transform, attributes = super(ROSAgentWrapper, self)._preprocess_sensor_spec(sensor_spec)
        new_type = self.SENSOR_TYPE_REMAPS.get(type_, type_)
        return new_type, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        for sensor_spec in self._agent.sensors():
            type_, id_, transform, attributes = self._preprocess_sensor_spec(sensor_spec)
            uid = self._agent.spawn_object(type_, id_, transform, attributes, attach_to=vehicle.id)
            self._sensors_list.append(uid)

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()

    def cleanup(self):
        for uid in self._sensors_list:
            self._agent.destroy_object(uid)
        self._sensors_list.clear()

        # Tick once to destroy the sensors
        CarlaDataProvider.get_world().tick()
