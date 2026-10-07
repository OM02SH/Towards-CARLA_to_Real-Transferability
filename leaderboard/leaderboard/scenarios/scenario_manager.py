#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import threading

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import PCLA, AgentWrapperFactory, AgentError, TickRuntimeError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

#from generator.online_carla_qa_gen import OnlineCarlaQAGenerator
#from leaderboard.leaderboard.autoagents.pcla_functions.QA_parser import ScenarioQA
#from leaderboard.leaderboard.autoagents.pcla_functions.QA_lidar import QALidar
from leaderboard.state_estimation.metrics_sensors import VehicleSensorSuite, SensorDataParser
from leaderboard.state_estimation.state_from_sensors import CarlaStateEstimatorEngine
from leaderboard.state_estimation.real_test_BBs import OnlineEvaluator
from leaderboard.state_estimation.real_test_ego import EgoStateOnlineEvaluator
from pathlib import Path
from leaderboard.state_estimation.shadow_criteria import ShadowCriteria
out_dir = Path("/home/omar-shaat/thesis_final/out")
out_dir.mkdir(parents=True, exist_ok=True)

class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0, client = None):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._watchdog = None
        self._agent_watchdog = None
        self._scenario_thread = None

        self._statistics_manager = statistics_manager
        self.client = client

        self.tick_count = 0

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)
        self.pcla = None
        self.counter = 0

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None
        
        self.sensor_suite = None
        self.parser = None
        self.bbs_evaluator = None
        self.ego_evaluator = None
        self.state_from_sensors = None
        self.sensor_criteria = None
        """if self.QALidar:
            self.QALidar.destroy()
            self.QALidar = None"""

    def load_scenario(self, scenario, agent_name, route_index, rep_number, config):
        """
        Load a new scenario
        """
        GameTime.restart()
        # self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        config.gps_route = scenario.gps_route
        config.route = scenario.route
        self.pcla = PCLA(agent_name, scenario.ego_vehicles[0], config, self.client)
        # self.QALidar = QALidar(CarlaDataProvider.get_world(), scenario.ego_vehicles[0])
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        # self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)

    def run_scenario(self):
        """
        Trigger the stdoesart of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        self._running = True

        # Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()

        self.world = CarlaDataProvider.get_world()
        map = self.world.get_map()
        self.sensor_suite = VehicleSensorSuite(self.world, self.ego_vehicles[0])
        self.parser = SensorDataParser()
        radar_transforms = {name: t for name, t in self.sensor_suite.mount_transforms.items() if "radar" in name}
        self.bbs_evaluator = OnlineEvaluator(world=self.world, ego_actor_id=self.ego_vehicles[0].id)
        self.ego_evaluator = EgoStateOnlineEvaluator(self.world, self.ego_vehicles[0].id)
        self.state_from_sensors = CarlaStateEstimatorEngine(self.ego_vehicles[0], self.world, map, self.sensor_suite.mount_transforms["lidar_slam"],
                                                            radar_transforms, self.ego_evaluator, self.bbs_evaluator)
        self.sensor_criteria = ShadowCriteria(self.ego_vehicles[0], self.scenario.route)
        #QAG = OnlineCarlaQAGenerator(world,map.name, self.scenario.ego_vehicles[0].id, map.name)
        while self._running:
            self._tick_scenario()
            self.sensor_criteria.update()
        #print("saving metrics")
        map_name = Path(map.name).name
        self.ego_evaluator.save_metrics_json(out_dir / f"ego_{map_name}_{self.counter}.json")
        self.bbs_evaluator.save_metrics_json(out_dir / f"bbs_{map_name}_{self.counter}.json")
        self.sensor_criteria.save_json(out_dir / f"criteria_{map_name}_{self.counter}.json")
        self.counter += 1

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self.tick_count += 1
            self._watchdog.pause()

            if self.tick_count > 4000:
                raise TickRuntimeError("RuntimeError, tick_count > 4000")

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                # ego_action = self._agent_wrapper()
                ego_action = self.pcla.get_action()
                #print("action")
                """if self.tick_count!=0 and self.tick_count/10 == 0:
                    lidar_points_ego = self.QALidar.get_points_ego_frame()
                    (combined_qas, num_questions, num_objects, num_questions_per_category,
                     key_object_infos, extra_flags) = QAG.QAGen(
                         CarlaDataProvider.get_world(), self._last_waypoints, lidar_points_ego)
                    qas = ScenarioQA(raw_qa_data= combined_qas)
                    Qs = qas.get_all_questions()
                    ego_action, waypoints, response = self.pcla.get_action(Qs)
                else:
                    ego_action, waypoints, response = self.pcla.get_action()"""
                #self._last_waypoints = waypoints
                sensors_data = self.parser.process_all_sensors(self.sensor_suite.get_latest_data())
                    # print(sensors_data)
                # update the state
                self.state_from_sensors.compute(self.world.get_snapshot().timestamp.delta_seconds,
                                                    self.ego_vehicles[0],sensors_data,self.tick_count)
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                self.compute_duration_time()
                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                          carla.Rotation(pitch=-90)))

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            try:
                from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
                world = CarlaDataProvider.get_world()
                if world is not None:
                    for sensor in world.get_actors().filter('*sensor*'):
                        if sensor.is_alive:
                            try:
                                sensor.stop()  # Safely close the active stream
                            except Exception:
                                pass
            except Exception:
                pass
            # --------------------------------------
            if self.scenario is not None:
                self.scenario.terminate()

            if self.pcla is not None:
                self.pcla.cleanup()
                self.pcla = None 
            if self.sensor_suite is not None:
                self.sensor_suite.destroy()
                self.sensor_suite = None
            self.analyze_scenario()

        if hasattr(self, 'sensor_suite') and self.sensor_suite is not None:
            try:
                if hasattr(self.sensor_suite, 'destroy'):
                    self.sensor_suite.destroy()
                elif hasattr(self.sensor_suite, 'cleanup'):
                    self.sensor_suite.cleanup()
            except Exception as e:
                print(f"Error cleaning up sensor suite: {e}")
            self.sensor_suite = None

        if hasattr(self, 'state_from_sensors') and self.state_from_sensors is not None:
            self.state_from_sensors = None

        # --------------------------------------------

        # Make sure the scenario thread finishes to avoid blocks
        self._running = False
        if self._scenario_thread is not None:
            self._scenario_thread.join(timeout=2.0)
            self._scenario_thread = None

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)
