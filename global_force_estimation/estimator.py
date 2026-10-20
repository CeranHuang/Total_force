from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import DTactDataset, IMAGE_PREPROCESSING_ID
from model.CNN import DensenetGlobal, ResnetGlobal


class Estimator:
    def __init__(self, cfg):
        self.device = torch.device(
            f"cuda:{cfg.get('cuda_index', 0)}" if torch.cuda.is_available() else "cpu"
        )
        model_name, model_layer = cfg["model_list"][cfg["model_choice"]]

        model_root = Path(cfg.get("save_dir", "saved_models"))
        if not model_root.is_absolute():
            model_root = PROJECT_ROOT / model_root
        checkpoints = sorted(model_root.glob("global_*/best.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No global checkpoint found under {model_root}; train a model with train_global.py first."
            )
        self.checkpoint_path = checkpoints[-1]

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if checkpoint.get("image_preprocessing") != IMAGE_PREPROCESSING_ID:
            raise ValueError(
                f"Unsupported checkpoint preprocessing: {checkpoint.get('image_preprocessing')!r}"
            )

        # Older global checkpoints recorded their architecture only in basic.txt.
        if "model_name" not in checkpoint or "model_layer" not in checkpoint:
            metadata_path = self.checkpoint_path.with_name("basic.txt")
            if metadata_path.is_file():
                for line in metadata_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("model:"):
                        label, _, depth = line.partition(":")[2].strip().rpartition("-")
                        legacy_names = {"ResNet": "Resnet", "DenseNet": "Densenet"}
                        if label in legacy_names and depth.isdecimal():
                            model_name, model_layer = legacy_names[label], int(depth)
                        break
        model_name = checkpoint.get("model_name", model_name)
        model_layer = checkpoint.get("model_layer", model_layer)
        builders = {"Resnet": ResnetGlobal, "Densenet": DensenetGlobal}
        if model_name not in builders:
            raise ValueError("Global-force model must be Resnet or Densenet.")
        self.model = builders[model_name](layer=int(model_layer), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        normalizer = checkpoint["normalizer"]
        self.force_mean = torch.as_tensor(
            normalizer["mean"], dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.force_std = torch.as_tensor(
            normalizer["std"], dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.image_type = checkpoint.get("image_type", cfg.get("image_type", "RGB"))
        self.image_size = checkpoint.get("input_size")
        print(f"Loaded global-force checkpoint: {self.checkpoint_path}")

    def predict_force(self, bgr_image):
        image = DTactDataset.preprocess_bgr_image(
            bgr_image, self.image_type, self.image_size
        )
        image = torch.from_numpy(image[None]).to(self.device)
        with torch.no_grad():
            force = self.model(image) * self.force_std + self.force_mean
        return force[0].cpu().numpy()
