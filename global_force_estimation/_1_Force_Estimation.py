from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml
from camera import Camera
from global_force_estimation.estimator import Estimator
from global_force_estimation.force_visualizer import Visualizer


def main():
    with (PROJECT_ROOT / "shape_config3.yaml").open("r", encoding="utf-8") as stream:
        camera_config = yaml.safe_load(stream)
    calibration_root = (
        PROJECT_ROOT / camera_config.get("calibration_root_dir", "IntrinsicCalibration")
    ).resolve()
    camera_config["calibration_root_dir"] = str(calibration_root)
    intrinsic = camera_config["intrinsic_calibration"]
    for key, default_name in (("camera_matrix_path", "camera_1_K.npy"),
                              ("dist_coeff_path", "camera_1_D.npy")):
        intrinsic[key] = str(calibration_root / Path(intrinsic.get(key) or default_name).name)

    with (Path(__file__).with_name("force_config.yaml")).open("r", encoding="utf-8") as stream:
        force_config = yaml.safe_load(stream)

    estimator = Estimator(force_config)
    with Camera(camera_config) as camera:
        visualizer = Visualizer()
        try:
            while camera.cap.isOpened():
                image = camera.get_rectify_crop_image()
                cv2.imshow("image", image)
                force = estimator.predict_force(image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if not visualizer.vis.poll_events():
                    break
                visualizer.update_force(force)
        finally:
            visualizer.vis.destroy_window()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()