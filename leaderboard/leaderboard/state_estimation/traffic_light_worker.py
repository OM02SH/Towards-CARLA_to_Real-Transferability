"""
Background worker that runs the fine-tuned YOLO stop/go traffic-light
classifier off the main CARLA tick thread. Holds a single-slot input
queue (only the latest pushed frame matters - stale frames are dropped,
not queued up) and a single-slot output queue of the latest result.

Usage:
    worker = TrafficLightWorker("weights/best.pt")
    worker.start()

    # each CARLA camera callback:
    worker.push_frame(carla_image)

    # each tick, wherever you populate CalculatedState:
    result = worker.should_stop()   # LightState.GO / LightState.STOP / LightState.NONE

    worker.stop()   # on shutdown
"""

import queue
import threading
from leaderboard.leaderboard.state_estimation.sensor_dataclasses import LightState

import numpy as np
from ultralytics import YOLO

STOP_CLASS_ID = 0  # matches data.yaml: 0=stop, 1=go
GO_CLASS_ID = 1

class TrafficLightWorker:
    def __init__(self, conf=0.4, device="cpu"):
        self._model = YOLO("best.pt")
        self._conf = conf
        self._device = device

        self._in_queue = queue.Queue(maxsize=1)
        self._out_queue = queue.Queue(maxsize=1)

        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def push_frame(self, carla_image):
        """
        Non-blocking. Converts the carla.Image to a numpy RGB array and
        drops it into the single-slot input queue, discarding whatever
        frame was waiting there (we only ever care about the latest one).
        """
        frame = self._carla_image_to_rgb(carla_image)
        self._drop_and_put(self._in_queue, frame)

    def should_stop(self):
        try:
            return self._out_queue.get_nowait()
        except queue.Empty:
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
        """
        Runs the model on one frame and returns a LightState for the
        single highest-confidence detected light. If multiple lights are
        visible in frame, this takes the most confident detection overall
        rather than e.g. the largest/closest one - revisit this policy if
        you find it picking a distant light over a directly-relevant one.
        """
        results = self._model.predict(frame, conf=self._conf, device=self._device, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return LightState.NONE

        best_idx = int(boxes.conf.argmax())
        best_cls = int(boxes.cls[best_idx])

        if best_cls == STOP_CLASS_ID:
            return LightState.STOP
        elif best_cls == GO_CLASS_ID:
            return LightState.GO
        return LightState.NONE

    @staticmethod
    def _carla_image_to_rgb(carla_image):
        """carla.Image.raw_data is BGRA, packed row-major."""
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        return array[:, :, :3][:, :, ::-1]  # drop alpha, BGR -> RGB

    @staticmethod
    def _drop_and_put(q, item):
        """Keep only the newest item in a maxsize=1 queue."""
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)