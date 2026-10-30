from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import DTactDataset, IMAGE_PREPROCESSING_ID
from model.CNN import GlobalForceHead, ResnetGlobal


class Estimator:
    def __init__(self, cfg):
        self.device = torch.device(
            f"cuda:{cfg.get('cuda_index', 0)}" if torch.cuda.is_available() else "cpu"
        )
        model_root = Path(cfg.get("save_dir", "saved_models"))
        if not model_root.is_absolute():
            model_root = PROJECT_ROOT / model_root
        checkpoints = sorted(model_root.glob("global_20261019_125052/best.pt"))
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

        if "model_name" in checkpoint:
            model_name = checkpoint["model_name"]
            model_layer = int(checkpoint["model_layer"])
        else:
            model_name, model_layer = cfg["model_list"][cfg["model_choice"]]
            model_layer = int(model_layer)
        if model_name == "Densenet" and model_layer == 169:
            backbone = models.densenet169(weights=None)
            backbone.classifier = torch.nn.Linear(backbone.classifier.in_features, 3)
            self.model = GlobalForceHead(backbone).to(self.device)
        elif model_name == "Resnet":
            self.model = ResnetGlobal(layer=model_layer, pretrained=False).to(self.device)
        else:
            raise ValueError(f"Unsupported global-force model: {model_name}-{model_layer}")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.input_size = checkpoint.get("input_size")

        normalizer = checkpoint["normalizer"]
        self.force_mean = torch.as_tensor(
            normalizer["mean"], dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.force_std = torch.as_tensor(
            normalizer["std"], dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.image_type = checkpoint.get("image_type", cfg.get("image_type", "RGB"))
        print(f"Loaded global-force checkpoint: {self.checkpoint_path}")

    def predict_force(self, bgr_image):
        if self.input_size is not None:
            height, width = self.input_size
            bgr_image = cv2.resize(bgr_image, (width, height))
        image = DTactDataset.preprocess_bgr_image(bgr_image, self.image_type)
        image = torch.from_numpy(image[None]).to(self.device)
        with torch.no_grad():
            force = self.model(image) * self.force_std + self.force_mean
        return force[0].cpu().numpy()
