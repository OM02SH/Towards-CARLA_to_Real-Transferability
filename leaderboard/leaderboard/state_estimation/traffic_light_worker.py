import queue
import threading
import numpy as np
from ultralytics import YOLO
from leaderboard.state_estimation.sensor_dataclasses import LightState
import leaderboard.state_estimation.hyperparams as hp
import os

class TrafficLightWorker:
    def __init__(self, conf=0.4, device="gpu"):
        self._model = YOLO(hp.YOLO_MODEL_PATH)
        self._conf = conf
        self._device = device
        self._in_queue = queue.Queue(maxsize=1)
        self._out_queue = queue.Queue(maxsize=1)
        self._thread = None
        self._running = False
        self._last_state = LightState.NONE
        self._hold_calls = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def push_frame(self, carla_image):
        frame = self._carla_image_to_rgb(carla_image)
        self._drop_and_put(self._in_queue, frame)

    def light_state(self):
        try:
            state, is_distorted = self._out_queue.get_nowait()
            if state != LightState.NONE:
                self._last_state = state
                if is_distorted:
                    self._hold_calls = hp.HOLD_MAX_CALLS
                else:
                    self._hold_calls = 0
                return state
            if self._hold_calls > 0:
                self._hold_calls -= 1
                return self._last_state
            return LightState.NONE
        except queue.Empty:
            if self._hold_calls > 0:
                self._hold_calls -= 1
                return self._last_state
            return LightState.NONE

    def _worker_loop(self):
        while self._running:
            try:
                frame = self._in_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            result = self._infer(frame)
            self._drop_and_put(self._out_queue, result)

    def _infer(self, frame):
        results = self._model.predict(frame, conf=self._conf, device=self._device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return LightState.NONE, False

        xyxy = boxes.xyxy.cpu().numpy()
        _, img_w = frame.shape[:2]
        
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights
        
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        center_distances = np.abs(centers_x - (img_w / 2.0)) / (img_w / 2.0)
        norm_areas = areas / (np.max(areas) + 1e-6)
        scores = norm_areas - (2.0 * center_distances)
        best_idx = int(np.argmax(scores))
        
        width = widths[best_idx]
        height = heights[best_idx]

        if height <= 0: return LightState.NONE, False
        best_cls = int(boxes.cls[best_idx])
        if best_cls == hp.STOP_CLASS_ID:
            state = LightState.STOP
        elif best_cls == hp.GO_CLASS_ID:
            state = LightState.GO
        else:
            state = LightState.NONE
            
        return state, (abs((width / height) - hp.EXPECTED_ASPECT_RATIO) / hp.EXPECTED_ASPECT_RATIO) > hp.ASPECT_RATIO_TOLERANCE

    @staticmethod
    def _carla_image_to_rgb(carla_image):
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        return array[:, :, :3][:, :, ::-1]

    @staticmethod
    def _drop_and_put(q, item):
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)
