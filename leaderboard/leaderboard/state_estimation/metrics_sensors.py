import carla
import queue
import logging
import numpy as np
import math

class VehicleSensorSuite:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world = world
        self.vehicle = vehicle
        self.bp_lib = world.get_blueprint_library()
        self.sensors = {}
        self.data_queues = {}
        bb = vehicle.bounding_box
        loc, ext = bb.location, bb.extent  
        margin = 0.08  

        self.surround_layout = {
            "front":    {"pos": (loc.x + ext.x + margin,  loc.y,                  loc.z + ext.z), "yaw":    0.0},
            "left" :    {"pos": (loc.x,                   loc.y - ext.y - margin, loc.z + ext.z), "yaw":  -90.0},
            "right":    {"pos": (loc.x,                   loc.y + ext.y + margin, loc.z + ext.z), "yaw":   90.0},
            "rear" :    {"pos": (loc.x - ext.x - margin,  loc.y,                  loc.z + ext.z), "yaw":  180.0},
        }
        self.lidar_pos = carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + ext.z + 0.2),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        self.mount_transforms = {}
        self._spawn_all_sensors()

    def _create_transform(self, pos: tuple, yaw: float) -> carla.Transform:
        """Helper to create a CARLA Transform from position and yaw angle."""
        location = carla.Location(x=pos[0], y=pos[1], z=pos[2])
        rotation = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        return carla.Transform(location, rotation)

    def _spawn_all_sensors(self):
        lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "80.0")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "2000000")
        lidar_bp.set_attribute("rotation_frequency", "10")
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-30.0")

        sensor_name = f"lidar"
        
        sensor = self.world.spawn_actor(lidar_bp, self.lidar_pos,
             attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        q = queue.Queue()
        sensor.listen(q.put)
        
        self.sensors[sensor_name] = sensor
        self.data_queues[sensor_name] = q
        self.mount_transforms[sensor_name] = self.lidar_pos  

        radar_bp = self.bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", "80")
        radar_bp.set_attribute("vertical_fov", "15")
        radar_bp.set_attribute("range", "200.0")
        radar_bp.set_attribute("points_per_second", "15000")

        s_radar_bp = self.bp_lib.find("sensor.other.radar")
        s_radar_bp.set_attribute("horizontal_fov", "100")
        s_radar_bp.set_attribute("vertical_fov", "20")
        s_radar_bp.set_attribute("range", "50.0")
        s_radar_bp.set_attribute("points_per_second", "10000")

        for name, cfg in self.surround_layout.items():
            sensor_name = f"radar_{name}"
            radar_pos = (cfg["pos"][0], cfg["pos"][1], 0.7)
            transform = self._create_transform(radar_pos, cfg["yaw"])
            if name in ["left","right"]:
                sensor = self.world.spawn_actor(s_radar_bp, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
            else:
                sensor = self.world.spawn_actor(radar_bp, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

            q = queue.Queue()
            sensor.listen(q.put)
            self.sensors[sensor_name] = sensor
            self.data_queues[sensor_name] = q
            self.mount_transforms[sensor_name] = transform
        
            
        imu_bp = self.bp_lib.find("sensor.other.imu")
        # Accelerometer Noise Standard Deviation (m/s²)
        """imu_bp.set_attribute('noise_accel_stddev_x', '0.15')
        imu_bp.set_attribute('noise_accel_stddev_y', '0.15')
        imu_bp.set_attribute('noise_accel_stddev_z', '0.15')

        # Gyroscope Noise Standard Deviation (rad/s)
        imu_bp.set_attribute('noise_gyro_stddev_x', '0.008')
        imu_bp.set_attribute('noise_gyro_stddev_y', '0.008')
        imu_bp.set_attribute('noise_gyro_stddev_z', '0.008')

        # Gyroscope Initial Bias (rad/s)
        imu_bp.set_attribute('noise_gyro_bias_x', '0.002')
        imu_bp.set_attribute('noise_gyro_bias_y', '0.002')
        imu_bp.set_attribute('noise_gyro_bias_z', '0.002')

        # Deterministic random seed
        imu_bp.set_attribute('noise_seed', '42')"""
        
        imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        
        imu_sensor = self.world.spawn_actor(
            imu_bp, imu_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        imu_q = queue.Queue()
        imu_sensor.listen(imu_q.put)
        
        self.sensors["imu"] = imu_sensor
        self.data_queues["imu"] = imu_q

        gnss_bp = self.bp_lib.find("sensor.other.gnss")
        """gnss_bp.set_attribute('noise_lat_stddev', '0.000018')
        gnss_bp.set_attribute('noise_lon_stddev', '0.000018')
        gnss_bp.set_attribute('noise_alt_stddev', '1')

        # Constant mean bias offset
        gnss_bp.set_attribute('noise_lat_bias', '0.000005')
        gnss_bp.set_attribute('noise_lon_bias', '0.000005')
        gnss_bp.set_attribute('noise_alt_bias', '2')

        # Deterministic random seed
        gnss_bp.set_attribute('noise_seed', '42')"""
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))

        gnss_sensor = self.world.spawn_actor(
            gnss_bp, gnss_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        gnss_q = queue.Queue()
        gnss_sensor.listen(gnss_q.put)

        self.sensors["gnss"] = gnss_sensor
        self.data_queues["gnss"] = gnss_q


    def get_latest_data(self, timeout: float = 1.0) -> dict:
        latest_data = {}
        for name, q in self.data_queues.items():
            try:
                item = q.get(block=True, timeout=timeout)  # wait for at least one
                while True:
                    try:
                        item = q.get_nowait()  # drain any backlog, keep newest
                    except queue.Empty:
                        break
                latest_data[name] = item
            except queue.Empty:
                latest_data[name] = None
        return latest_data

    def destroy(self):
        for name, sensor in self.sensors.items():
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        self.data_queues.clear()

class SensorDataParser:
    def process_all_sensors(self, raw_data_dict: dict) -> dict:
        processed_data = {
            "lidar": None,
            "radars": {},
            "imu": None,
            "gnss": None
        }
        for name, raw_measurement in raw_data_dict.items():
            if name.startswith("lidar"):
                processed_data["lidar"] = raw_measurement
            elif name.startswith("radar_"):
                processed_data["radars"][name] = raw_measurement
            elif name == "imu":
                processed_data["imu"] = raw_measurement
            elif name == "gnss":
                processed_data["gnss"] = raw_measurement
        return processed_data