#!/usr/bin/env python3
import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ============================================================
# STATIC PATH: your CarlaOccV1 dataset
# ============================================================
DATA_ROOT = Path("/bulk/users/omar-shaat/CarlaOccV1").resolve()

CALIB_FILE = DATA_ROOT / "calib" / "calib.yaml"

# Existing files produced by your TransFusion-L conversion
INFO_FILES = {
    "train": DATA_ROOT / "carlaocc_infos_train.pkl",
    "val": DATA_ROOT / "carlaocc_infos_val.pkl",
    "test": DATA_ROOT / "carlaocc_infos_test.pkl",
}

# Only these 6 cameras are used by BEVFusion.
CAMERAS = [f"cam_{i:02d}" for i in range(6)]


def load_yaml(path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_infos(path):
    with path.open("rb") as f:
        infos = pickle.load(f)

    if not isinstance(infos, list):
        raise TypeError(f"{path} does not contain a list of frame infos")

    return infos


def rotation_to_wxyz(R):
    """
    scipy: [x, y, z, w]
    OpenPCDet/NuScenes-style Quaternion: [w, x, y, z]
    """
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.asarray(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=np.float64,
    )


def build_camera_calibration(calib):
    """
    Read the fixed CarlaOcc camera calibration.

    IMPORTANT:
    The transforms are copied from calib.yaml as provided.
    No RGB files and no LiDAR files are copied or regenerated.

    The CARLA camera-coordinate conversion needed for projection is
    intentionally NOT baked into sensor2lidar_rotation here. It belongs
    in the OpenPCDet camera loader if your BEVFusion CARLA loader uses the
    standard CARLA xyz -> y,-z,x conversion.
    """
    K = np.asarray(
        calib["cam_settings"]["intrinsics"],
        dtype=np.float64,
    )

    camera_calibs = {}

    for cam_name in CAMERAS:
        T = np.asarray(
            calib["sensors"][cam_name]["transform"],
            dtype=np.float64,
        )

        if T.shape != (4, 4):
            raise ValueError(
                f"{cam_name} transform has shape {T.shape}, expected (4,4)"
            )

        R = T[:3, :3]
        t = T[:3, 3]

        camera_calibs[cam_name] = {
            "sensor2lidar_rotation": R,
            "sensor2lidar_translation": t,
            "sensor2ego_rotation": rotation_to_wxyz(R),
            "sensor2ego_translation": t.copy(),

            # OpenPCDet uses this name.
            "camera_intrinsics": K.copy(),

            # Some BEVFusion/NuScenes variants use this spelling.
            "cam_intrinsic": K.copy(),
        }

    return camera_calibs


def make_frame_info(info, camera_calibs):
    info = dict(info)

    frame_id = str(info["frame_id"])

    # Your existing frame IDs are e.g. Town01_Opt_Seq00_0000.
    seq_name, frame_str = frame_id.rsplit("_", 1)
    frame = int(frame_str)

    lidar_path = DATA_ROOT / "converted_lidar" / seq_name / f"{frame:04d}.bin"

    if not lidar_path.exists():
        raise FileNotFoundError(f"Missing LiDAR: {lidar_path}")

    # Absolute path: OpenPCDet can use it directly and nothing is copied.
    info["lidar_path"] = str(lidar_path)

    cams = {}

    for cam_id, cam_name in enumerate(CAMERAS):
        image_path = (
            DATA_ROOT
            / "all_rgb"
            / seq_name
            / "rgb"
            / f"image_{cam_id:02d}"
            / f"{frame:04d}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        cam = dict(camera_calibs[cam_name])

        # Absolute path to the EXISTING PNG.
        # No image is copied, resized, converted or duplicated.
        cam["data_path"] = str(image_path)

        cams[cam_name] = cam

    info["cams"] = cams

    return info


def convert_split(split, camera_calibs):
    src = INFO_FILES[split]
    dst = DATA_ROOT / f"bevfusion_infos_{split}.pkl"

    if not src.exists():
        raise FileNotFoundError(src)

    infos = load_infos(src)

    converted = []

    for i, info in enumerate(infos, 1):
        converted.append(make_frame_info(info, camera_calibs))

        if i % 500 == 0 or i == len(infos):
            print(f"{split}: {i}/{len(infos)}")

    # ONLY this new .pkl file is written.
    with dst.open("wb") as f:
        pickle.dump(
            converted,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Created: {dst}")
    return len(converted)


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT)

    if not CALIB_FILE.exists():
        raise FileNotFoundError(CALIB_FILE)

    print(f"Dataset: {DATA_ROOT}")
    print(f"Calibration: {CALIB_FILE}")

    calib = load_yaml(CALIB_FILE)
    camera_calibs = build_camera_calibration(calib)

    totals = {}

    for split in ("train", "val", "test"):
        totals[split] = convert_split(split, camera_calibs)

    print("\nDone:")
    for split, n in totals.items():
        print(f"  {split}: {n} frames")


if __name__ == "__main__":
    main()
