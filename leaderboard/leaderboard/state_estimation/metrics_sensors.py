import carla
import queue
import numpy as np
import cv2

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
        cam_bp_standard = self.create_camera_bp(self.bp_lib, 1408, 376, 103.744808)
        self.camera_configs = {
            "cam_00": { # Front-left
                "loc": carla.Location(x=0.8, y=-0.3, z=-0.18),
                "rot": carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                "bp": cam_bp_standard
            },
            "cam_01": { # Front-right
                "loc": carla.Location(x=0.8, y=0.3, z=-0.18),
                "rot": carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                "bp": cam_bp_standard
            },
            "cam_02": { # Left 
                "loc": carla.Location(x=0.0, y=-0.46, z=-0.18),
                "rot": carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0), # Pointing CARLA Left
                "bp": cam_bp_standard
            },
            "cam_03": { # Right 
                "loc": carla.Location(x=0.0, y=0.46, z=-0.18),
                "rot": carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0), # Pointing CARLA Right
                "bp": cam_bp_standard
            },
            "cam_04": { # Rear-left
                "loc": carla.Location(x=-0.8, y=-0.3, z=-0.18),
                "rot": carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0), # Pointing Backward
                "bp": cam_bp_standard
            },
            "cam_05": { # Rear-right
                "loc": carla.Location(x=-0.8, y=0.3, z=-0.18),
                "rot": carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0), # Pointing Backward
                "bp": cam_bp_standard
            }
        }
        self.mount_transforms = {}
        self._spawn_all_sensors()
    def create_camera_bp(self, bp_lib, width, height, fov):
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", str(fov))
        return camera_bp
    def _create_transform(self, pos: tuple, yaw: float) -> carla.Transform:
        """Helper to create a CARLA Transform from position and yaw angle."""
        location = carla.Location(x=pos[0], y=pos[1], z=pos[2])
        rotation = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        return carla.Transform(location, rotation)

    def _spawn_all_sensors(self):

        for cam_name, config in self.camera_configs.items():
            cam_loc = carla.Location(
                x = self.lidar_pos.location.x + config["loc"].x,
                y = self.lidar_pos.location.y + config["loc"].y,
                z = self.lidar_pos.location.z + config["loc"].z
            )
            cam_rot = carla.Rotation(
                pitch = self.lidar_pos.rotation.pitch + config["rot"].pitch,
                yaw   = self.lidar_pos.rotation.yaw + config["rot"].yaw,
                roll  = self.lidar_pos.rotation.roll + config["rot"].roll
            )
            transform = carla.Transform(cam_loc, cam_rot)
            sensor = self.world.spawn_actor(config["bp"], transform, 
                attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
            
            q = queue.Queue()
            sensor.listen(q.put)
            self.sensors[cam_name] = sensor
            self.data_queues[cam_name] = q
            self.mount_transforms[cam_name] = transform

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
        
        imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
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
    YOLO_CAMS = ("cam_00", "cam_01")

    def process_all_sensors(self, raw_data_dict: dict) -> dict:
        processed_data = {
            "lidar": None,
            "cameras": {},
            "yolo_cameras": None,
            "radars": {},
            "imu": None,
            "gnss": None
        }
        yolo_frames = {}
        for name, raw_measurement in raw_data_dict.items():
            if raw_measurement is None:
                continue
            elif name.startswith("lidar"):
                processed_data["lidar"] = raw_measurement
            elif name.startswith("cam_"):
                img_bgra = np.frombuffer(raw_measurement.raw_data, dtype=np.uint8)
                img_bgra = img_bgra.reshape((raw_measurement.height, raw_measurement.width, 4))
                img_bgr = img_bgra[:, :, :3]
                img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
                processed_data["cameras"][name] = cv2.resize(img_rgb, (704, 192))
                if name in self.YOLO_CAMS:
                    yolo_frames[name] = img_bgr
            elif name.startswith("radar_"):
                processed_data["radars"][name] = raw_measurement
            elif name == "imu":
                processed_data["imu"] = raw_measurement
            elif name == "gnss":
                processed_data["gnss"] = raw_measurement
        frames = [yolo_frames[n] for n in self.YOLO_CAMS if n in yolo_frames]
        if frames:
            processed_data["yolo_camera"] = np.ascontiguousarray(np.vstack(frames))
        if len(processed_data["cameras"]) == 6:
            cam_order = ["cam_00", "cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]
            stacked_imgs = np.stack([processed_data["cameras"][cam] for cam in cam_order], axis=0)
            processed_data["stacked_cameras"] = stacked_imgs
            for name in  ["cam_00", "cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]:
                img_to_save_bgr = cv2.cvtColor(processed_data["cameras"][name], cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"/app/output/{name}_output.png", img_to_save_bgr)
                print("one saved")
        return processed_data