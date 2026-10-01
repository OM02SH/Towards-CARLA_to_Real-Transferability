import numpy as np
import carla

class QALidar:
    """
    Privileged LiDAR for GT annotation only — never exposed to the driving
    agent. Mirrors the exact mount Bench2Drive-VL's own dataset used.
    """
    MOUNT_POS = np.array([-0.39, 0.0, 1.84])   # x, y, z relative to ego
    MOUNT_YAW = 0.0                             # degrees

    def __init__(self, world, ego_vehicle):
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '85')
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('channels', '64')
        bp.set_attribute('points_per_second', '600000')
        bp.set_attribute('dropoff_general_rate', '0.0')
        bp.set_attribute('dropoff_intensity_limit', '0.0')
        bp.set_attribute('dropoff_zero_intensity', '0.0')

        transform = carla.Transform(
            carla.Location(x=-0.39, y=0.0, z=1.84),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
        )
        self.sensor = world.spawn_actor(bp, transform, attach_to=ego_vehicle)
        self._latest = None
        self.sensor.listen(self._callback)

    def _callback(self, measurement):
        pts = np.frombuffer(measurement.raw_data, dtype=np.float32)
        self._latest = pts.reshape(-1, 4)[:, :3]
        self._latest_frame = measurement.frame

    def get_points_ego_frame(self):
        if self._latest is None:
            return None
        yaw = np.deg2rad(self.MOUNT_YAW)
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw),  np.cos(yaw), 0.0],
                         [0.0,          0.0,         1.0]])
        return (rot @ self._latest.T).T + self.MOUNT_POS

    def destroy(self):
        self.sensor.destroy()