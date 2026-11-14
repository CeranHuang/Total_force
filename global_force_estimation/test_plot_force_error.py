"""Offline error-plot checks, including a tiny training-completion run."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from global_force_estimation.plot_force_error import main, save_force_error_plot


class ForceErrorPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def rows(self):
        with (self.root / "plots" / "force_error_by_range.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_signed_errors_grouped_by_truth_and_magnitude_is_norm_difference(self):
        truth = np.array([[3., 4., -12.], [3., 4., -12.]])
        prediction = np.array([[0., 0., -5.], [0., 0., -5.]])
        path = save_force_error_plot(prediction, truth, self.root / "plots")
        self.assertTrue(path.is_file())
        rows = [row for row in self.rows() if int(row["count"])]
        self.assertEqual([row["component"] for row in rows], ["Fx", "Fy", "Fz", "Magnitude"])
        self.assertEqual([float(row["bias_n"]) for row in rows], [-3., -4., 7., -8.])
        self.assertEqual([float(row["mae_n"]) for row in rows], [3., 4., 7., 8.])
        self.assertEqual(float(rows[0]["lower_n"]), 1.5)
        self.assertEqual(float(rows[-1]["lower_n"]), 7.5)

    def test_boundaries_and_outliers_are_counted_once(self):
        truth = np.array([
            [-100., -100., -100.], [-1.5, -1.5, -7.5],
            [0., 0., -5.], [1.5, 1.5, -2.5], [100., 100., 100.],
        ])
        save_force_error_plot(truth, truth, self.root / "plots")
        for component in ("Fx", "Fy", "Fz"):
            rows = [row for row in self.rows() if row["component"] == component]
            self.assertEqual([int(row["count"]) for row in rows], [1, 1, 1, 2])
            self.assertTrue(all(float(row["rmse_n"]) == 0 for row in rows))
        self.assertEqual(sum(int(row["count"]) for row in self.rows()), 20)

    def test_mixed_signed_errors_keep_bias_distinct_from_mae(self):
        truth = np.zeros((2, 3))
        prediction = np.array([[-1., 0., 0.], [3., 0., 0.]])
        save_force_error_plot(prediction, truth, self.root / "plots")
        row = next(row for row in self.rows() if row["component"] == "Fx" and int(row["count"]))
        self.assertEqual(float(row["bias_n"]), 1.)
        self.assertEqual(float(row["median_n"]), 1.)
        self.assertEqual(float(row["mae_n"]), 2.)
        self.assertAlmostEqual(float(row["rmse_n"]), np.sqrt(5.))

    def test_singleton_empty_and_constant_groups_render(self):
        for count in (1, 24):
            with self.subTest(count=count):
                values = np.zeros((count, 3))
                save_force_error_plot(values, values, self.root / "plots")
                self.assertEqual(sum(int(row["count"]) for row in self.rows()), 4 * count)

    def test_invalid_inputs_rejected_before_output(self):
        for prediction, truth in (
            (np.zeros((2, 3)), np.zeros((1, 3))),
            (np.zeros((2, 4, 4, 3)), np.zeros((2, 4, 4, 3))),
            (np.empty((0, 3)), np.empty((0, 3))),
            (np.array([[0., 0., np.nan]]), np.zeros((1, 3))),
            (np.zeros((1, 3)), np.array([[0., np.inf, 0.]])),
        ):
            with self.subTest(shape=prediction.shape), self.assertRaises(ValueError):
                save_force_error_plot(prediction, truth, self.root / "plots")
        self.assertFalse((self.root / "plots").exists())

    def test_cli_reads_saved_arrays(self):
        np.save(self.root / "test_predictions.npy", np.ones((80, 3)))
        np.save(self.root / "test_targets.npy", np.linspace(-1., 1., 240).reshape(80, 3))
        main([str(self.root)])
        self.assertTrue((self.root / "plots" / "force_error_distribution.png").is_file())

    def test_training_completion_saves_denormalized_plot_data(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from global_force_estimation.train_global import ForceNormalizer, GlobalForceTrainer

        trainer = GlobalForceTrainer.__new__(GlobalForceTrainer)
        trainer.device = torch.device("cpu")
        trainer.save_dir = self.root
        trainer.epochs = 1
        trainer.normalizer = ForceNormalizer(mean=[0., 0., -3.], std=[2., 3., 4.])
        trainer.model = torch.nn.Linear(3, 3)
        with torch.no_grad():
            trainer.model.weight.zero_()
            trainer.model.bias.zero_()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
        trainer.criterion = torch.nn.SmoothL1Loss()
        dataset = TensorDataset(torch.zeros((4, 3)), torch.tensor([[0., 0., -3.]]).repeat(4, 1))
        dataset.image_type = "RGB"
        trainer.train_dataset = dataset
        trainer.train_loader = trainer.validation_loader = trainer.test_loader = DataLoader(dataset, batch_size=2)
        trainer.train()
        np.testing.assert_array_equal(np.load(self.root / "test_predictions.npy"), [[0., 0., -3.]] * 4)
        self.assertTrue((self.root / "best.pt").is_file())
        self.assertTrue((self.root / "test_metrics.txt").is_file())
        self.assertTrue((self.root / "plots" / "force_error_distribution.png").is_file())
        self.assertEqual(sum(int(row["count"]) for row in self.rows()), 16)


if __name__ == "__main__":
    unittest.main()
