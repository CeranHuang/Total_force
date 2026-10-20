"""Train an image-to-global-force regressor from train/validation/test CSVs.

Each sample is ``image -> [Fx, Fy, Fz]``.  The target is standardized with
statistics fitted solely on the training split; predictions remain signed, so
there is deliberately no ReLU or output clipping.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import DTactDataset, IMAGE_PREPROCESSING_ID
from model.CNN import DensenetGlobal, ResnetGlobal


AXIS_NAMES = ("Fx", "Fy", "Fz")


class GlobalForceDataset(Dataset):
    """Read image paths and force labels from one precomputed split CSV."""

    def __init__(self, root_path, image_type="RGB", image_size=None, augmentation=None, mode="train"):
        if mode not in ("train", "validation", "test"):
            raise ValueError("mode must be 'train', 'validation', or 'test'.")
        self.root_path = Path(root_path).resolve()
        self.mode = mode
        self.csv_path = self.root_path / f"{mode}.csv"
        self.image_type = image_type
        self.image_size = None if image_size is None else tuple(int(value) for value in image_size)
        self.augmentation = DTactDataset._validate_augmentation_config(augmentation)
        self._augmentation_rng = None
        self._augmentation_worker_id = None
        self.samples = []
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Missing split CSV: {self.csv_path}. "
                "Set data_dir to the directory containing train.csv, validation.csv and test.csv."
            )
        with self.csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = {"image_path", "force_fx", "force_fy", "force_fz"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.csv_path}: missing columns {sorted(missing)}.")
            for row_number, row in enumerate(reader, start=2):
                stored_path = (row["image_path"] or "").strip()
                if not stored_path:
                    raise ValueError(f"{self.csv_path}:{row_number}: image_path must not be empty.")
                image_path = DTactDataset._resolve_dataset_path(self, stored_path)
                try:
                    force = np.asarray([float(row[f"force_{axis}"]) for axis in ("fx", "fy", "fz")],
                                       dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{self.csv_path}:{row_number}: invalid force values.") from exc
                if not np.isfinite(force).all():
                    raise ValueError(f"{self.csv_path}:{row_number}: force must be finite.")
                self.samples.append((str(image_path), force))
        if not self.samples:
            raise ValueError(f"No samples found in split CSV: {self.csv_path}.")

    def __len__(self):
        return len(self.samples)

    def _get_augmentation_rng(self):
        return DTactDataset._get_augmentation_rng(self)

    def get_sample(self, index, augment=False):
        path, force = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        if augment:
            image = DTactDataset.augment_bgr_image(self, image)
        return DTactDataset.preprocess_bgr_image(
            image, self.image_type, self.image_size
        ), force, path

    def __getitem__(self, index):
        return self.get_sample(index, augment=self.mode == "train")


class GlobalForceSubset(Dataset):
    """Select samples and optionally augment only this subset."""

    def __init__(self, dataset, indices, augment=False):
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset.get_sample(
            self.indices[index], augment=self.augment
        )


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def unpack_batch(batch):
    if len(batch) < 2:
        raise ValueError("Dataset must return (image, force[, image_path]).")
    return batch[0], batch[1], batch[2] if len(batch) > 2 else None


def as_global_force(labels):
    """Validate and return batch labels as float32 ``[B, 3]``."""
    labels = torch.as_tensor(labels, dtype=torch.float32)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(
            "Global force labels must have shape [B, 3]; "
            f"got {tuple(labels.shape)}."
        )
    if not torch.isfinite(labels).all():
        raise ValueError("Global force labels must be finite.")
    return labels


class ForceNormalizer:
    """Per-axis normalizer saved in every checkpoint for reproducible inference."""

    def __init__(self, mean=None, std=None, eps=1e-6):
        self.mean = None if mean is None else torch.as_tensor(mean).reshape(1, 3)
        self.std = None if std is None else torch.as_tensor(std).reshape(1, 3)
        self.eps = eps

    def fit(self, loader):
        total = torch.zeros(3, dtype=torch.float64)
        total_sq = torch.zeros(3, dtype=torch.float64)
        count = 0
        for batch in loader:
            _, labels, _ = unpack_batch(batch)
            labels = as_global_force(labels).double()
            total += labels.sum(dim=0)
            total_sq += labels.square().sum(dim=0)
            count += labels.shape[0]
        if count == 0:
            raise ValueError("Training split is empty.")
        mean = total / count
        variance = (total_sq / count - mean.square()).clamp_min(self.eps ** 2)
        self.mean = mean.float().reshape(1, 3)
        self.std = variance.sqrt().float().reshape(1, 3)
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, values):
        return (values - self.mean) / self.std

    def denormalize(self, values):
        return values * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])


class GlobalForceMetrics:
    def __init__(self, target_std):
        self.count = 0
        self.abs_sum = torch.zeros(3, dtype=torch.float64)
        self.square_sum = torch.zeros(3, dtype=torch.float64)
        self.target_std = torch.as_tensor(target_std, dtype=torch.float64).cpu().reshape(3)
        if not torch.isfinite(self.target_std).all() or torch.any(self.target_std <= 0.0):
            raise ValueError("target_std must contain three finite positive values.")

    def update(self, prediction, target):
        error = (prediction.detach().cpu() - target.detach().cpu()).double()
        self.abs_sum += error.abs().sum(dim=0)
        self.square_sum += error.square().sum(dim=0)
        self.count += error.shape[0]

    def compute(self):
        if self.count == 0:
            raise ValueError("No samples available for metric calculation.")
        mae = self.abs_sum / self.count
        return {
            "mae": mae.float(),
            "rmse": torch.sqrt(self.square_sum / self.count).float(),
            "nmae": (mae / self.target_std).float(),
        }


class GlobalForceTrainer:
    def __init__(self, cfg, config_dir, cli_args=None):
        args = self._parse_args(cli_args)
        self.cfg = cfg
        self.args = args
        self.pretrained = (
            cfg.get("pretrained", True)
            if args.pretrained is None
            else args.pretrained
        )
        self.device = torch.device(
            f"cuda:{args.cuda_index if args.cuda_index is not None else cfg.get('cuda_index', 0)}"
            if torch.cuda.is_available() else "cpu"
        )
        data_dir = self._resolve_path(args.data_dir or cfg.get("data_dir", "Dataset"), PROJECT_ROOT)
        self.data_dir = data_dir
        self.batch_size = args.batch_size or int(cfg.get("batch_size", 32))
        dataset_kwargs = {
            "root_path": data_dir,
            "image_type": args.image_type or cfg.get("image_type", "RGB"),
            "image_size": cfg.get("img_size"),
            "augmentation": cfg.get("augmentation"),
        }
        self.train_dataset = GlobalForceDataset(mode="train", **dataset_kwargs)
        self.validation_dataset = GlobalForceDataset(mode="validation", **dataset_kwargs)
        self.test_dataset = GlobalForceDataset(mode="test", **dataset_kwargs)
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=False)

        stats_dataset = GlobalForceSubset(self.train_dataset, range(len(self.train_dataset)))
        stats_loader = DataLoader(stats_dataset, self.batch_size, shuffle=False)
        self.normalizer = ForceNormalizer().fit(stats_loader).to(self.device)
        self.model_name, configured_layer = self._config_model_spec(cfg)
        self.model_layer = args.model_layer if args.model_layer is not None else configured_layer
        builder = {"Resnet": ResnetGlobal, "Densenet": DensenetGlobal}[self.model_name]
        self.model = builder(layer=self.model_layer, pretrained=self.pretrained).to(self.device)
        self.criterion = nn.SmoothL1Loss()
        learning_rate = args.learning_rate or float(cfg.get("learning_rate", 1e-3))
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate,
            weight_decay=args.weight_decay if args.weight_decay is not None else float(cfg.get("weight_decay", 0.0)),
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)
        self.epochs = args.num_epoch or int(cfg.get("num_epoch", 100))
        timestamp = time.strftime("global_%Y%m%d_%H%M%S")
        self.save_dir = self._resolve_path(cfg.get("save_dir", "saved_models"), PROJECT_ROOT) / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=False)
        self._write_metadata(learning_rate, self.model_layer)

    @staticmethod
    def _resolve_path(value, base_dir):
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path).resolve()

    @staticmethod
    def _config_model_spec(cfg):
        name, layer = cfg.get("model_list", [["Resnet", 18]])[cfg.get("model_choice", 0)]
        if name not in ("Resnet", "Densenet"):
            raise ValueError("Global-force model must be Resnet or Densenet.")
        return name, int(layer)

    @staticmethod
    def _parse_args(cli_args):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--config", type=Path)
        parser.add_argument("--data-dir", type=Path,
                            help="Directory containing train.csv, validation.csv and test.csv")
        parser.add_argument("--model-layer", type=int)
        parser.add_argument("--image-type", choices=("RGB", "GRAY"))
        parser.add_argument("--cuda-index", type=int)
        parser.add_argument("--pretrained", type=str2bool, default=None)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--num-epoch", type=int)
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--weight-decay", type=float)
        return parser.parse_args(cli_args)

    def _prepare_batch(self, batch):
        images, labels, paths = unpack_batch(batch)
        images = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        labels = as_global_force(labels).to(self.device)
        return images, labels, paths

    def _write_metadata(self, learning_rate, layer):
        model_label = {"Resnet": "ResNet", "Densenet": "DenseNet"}[self.model_name]
        (self.save_dir / "basic.txt").write_text(
            "task: image_to_global_force\n"
            f"model: {model_label}-{layer}\naxes: {','.join(AXIS_NAMES)}\n"
            f"pretrained: {self.pretrained}\nbatch_size: {self.batch_size}\n"
            f"epochs: {self.epochs}\nlearning_rate: {learning_rate}\n"
            f"train_samples: {len(self.train_dataset)}\n"
            f"validation_samples: {len(self.validation_dataset)}\n"
            f"test_samples: {len(self.test_dataset)}\n"
            f"data_dir: {self.data_dir}\n"
            f"image_size: {self.train_dataset.image_size}\n"
            f"train_augmentation_enabled: {self.train_dataset.augmentation['enabled']}\n"
            "labels: raw collection sensor axes; no force rotation or sign reversal\n"
            "split: train.csv / validation.csv / test.csv (precomputed; no resplitting)\n"
            f"image_preprocessing: {IMAGE_PREPROCESSING_ID}\n"
            f"target_mean: {self.normalizer.mean.flatten().tolist()}\n"
            f"target_std: {self.normalizer.std.flatten().tolist()}\n",
            encoding="utf-8",
        )
        for mode, dataset in (("train", self.train_dataset),
                              ("validation", self.validation_dataset), ("test", self.test_dataset)):
            np.save(self.save_dir / f"{mode}_image_paths.npy",
                    np.asarray([path for path, _ in dataset.samples]))

    @staticmethod
    def _format_metrics(metrics):
        return (
            f"MAE(Fx,Fy,Fz)={np.round(metrics['mae'].numpy(), 5)}, "
            f"RMSE(Fx,Fy,Fz)={np.round(metrics['rmse'].numpy(), 5)}, "
            f"NMAE(Fx,Fy,Fz)={np.round(metrics['nmae'].numpy(), 5)}"
        )

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        count = 0
        metrics = GlobalForceMetrics(self.normalizer.std)
        with torch.no_grad():
            for batch in loader:
                images, labels, _ = self._prepare_batch(batch)
                prediction_normalized = self.model(images)
                loss = self.criterion(prediction_normalized, self.normalizer.normalize(labels))
                total_loss += loss.item() * images.shape[0]
                count += images.shape[0]
                metrics.update(self.normalizer.denormalize(prediction_normalized), labels)
        return total_loss / count, metrics.compute()

    def _save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "model_name": self.model_name,
            "model_layer": self.model_layer,
            "model_state_dict": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "axis_order": AXIS_NAMES,
            "pretrained_input_normalization": True,
            "image_preprocessing": IMAGE_PREPROCESSING_ID,
            "image_type": self.train_dataset.image_type,
            "input_size": self.train_dataset.image_size,
            "force_frame": "raw_collection_sensor",
        }, self.save_dir / "best.pt")

    def train(self):
        writer = SummaryWriter(log_dir=str(self.save_dir / "log"))
        best_loss = float("inf")
        history = {
            "epoch": [],
            "loss/train": [],
            "loss/validation": [],
            **{
                metric: {axis: [] for axis in AXIS_NAMES}
                for metric in ("mae", "rmse", "nmae")
            },
        }
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            metrics = GlobalForceMetrics(self.normalizer.std)
            for batch in self.train_loader:
                images, labels, _ = self._prepare_batch(batch)
                self.optimizer.zero_grad()
                normalized_prediction = self.model(images)
                loss = self.criterion(normalized_prediction, self.normalizer.normalize(labels))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.shape[0]
                metrics.update(self.normalizer.denormalize(normalized_prediction), labels)
            train_loss = running_loss / len(self.train_dataset)
            validation_loss, validation_metrics = self._evaluate(self.validation_loader)
            self.scheduler.step(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                self._save_checkpoint(epoch)
            train_metrics = metrics.compute()
            print(f"Epoch {epoch}/{self.epochs} | train={train_loss:.6f} | val={validation_loss:.6f}\n"
                  f"  train: {self._format_metrics(train_metrics)}\n"
                  f"  val:   {self._format_metrics(validation_metrics)}")
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", validation_loss, epoch)
            history["epoch"].append(epoch)
            history["loss/train"].append(train_loss)
            history["loss/validation"].append(validation_loss)
            for index, axis in enumerate(AXIS_NAMES):
                writer.add_scalar(f"mae/{axis}_validation", validation_metrics["mae"][index], epoch)
                writer.add_scalar(f"rmse/{axis}_validation", validation_metrics["rmse"][index], epoch)
                writer.add_scalar(f"nmae/{axis}_validation", validation_metrics["nmae"][index], epoch)
                for metric in ("mae", "rmse", "nmae"):
                    history[metric][axis].append(
                        float(validation_metrics[metric][index])
                    )
        writer.close()
        self._save_training_plots(history)
        self.load_best_checkpoint()
        self.test_record()

    def _save_training_plots(self, history):
        plot_dir = self.save_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        epochs = history["epoch"]

        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(epochs, history["loss/train"], label="Train")
        axis.plot(epochs, history["loss/validation"], label="Validation")
        axis.set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Smooth L1 loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / "loss_curves.png", dpi=150)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        metric_labels = (("mae", "MAE (N)"), ("rmse", "RMSE (N)"), ("nmae", "NMAE"))
        for axis, (metric, ylabel) in zip(axes, metric_labels):
            for force_axis in AXIS_NAMES:
                axis.plot(epochs, history[metric][force_axis], label=force_axis)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend()
        axes[0].set_title("Validation Metrics")
        axes[-1].set_xlabel("Epoch")
        figure.tight_layout()
        figure.savefig(plot_dir / "validation_metrics.png", dpi=150)
        plt.close(figure)

    def load_best_checkpoint(self):
        checkpoint = torch.load(self.save_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.normalizer = ForceNormalizer.from_state_dict(checkpoint["normalizer"]).to(self.device)
        return checkpoint

    def test_record(self):
        self.model.eval()
        predictions, targets, image_paths = [], [], []
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels, paths = self._prepare_batch(batch)
                predictions.append(self.normalizer.denormalize(self.model(images)).cpu().numpy())
                targets.append(labels.cpu().numpy())
                if paths is not None:
                    image_paths.extend(str(path) for path in paths)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        metrics = GlobalForceMetrics(self.normalizer.std)
        metrics.update(torch.from_numpy(predictions), torch.from_numpy(targets))
        np.save(self.save_dir / "test_predictions.npy", predictions)
        np.save(self.save_dir / "test_targets.npy", targets)
        if image_paths:
            np.save(self.save_dir / "test_image_paths.npy", np.asarray(image_paths))
        summary = self._format_metrics(metrics.compute())
        (self.save_dir / "test_metrics.txt").write_text(summary + "\n", encoding="utf-8")
        print(f"Test: {summary}")


def main(cli_args=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(cli_args)
    config_path = known.config or Path(__file__).with_name("force_config.yaml")
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    trainer = GlobalForceTrainer(cfg, config_path.resolve().parent, cli_args)
    trainer.train()


if __name__ == "__main__":
    main()
