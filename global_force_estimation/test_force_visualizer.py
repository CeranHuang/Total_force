"""Offline checks; no camera, model checkpoint or desktop windows required."""
from unittest.mock import Mock

import numpy as np
import pytest

from global_force_estimation import force_visualizer as module


@pytest.fixture
def display(monkeypatch):
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(module.cv2, name, Mock())
    monkeypatch.setattr(module.cv2, "getWindowProperty", lambda *args: 1)
    visualizer = module.DashboardVisualizer(force_threshold_n=(0.5, 0.5, 0.5))
    yield visualizer
    visualizer.close()


def test_camera_and_signed_forces_share_one_panel(display, monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    force = np.array([0.2, -2.0, -5.0])
    display.update_force(force, image=image)
    np.testing.assert_array_equal(display.force_history[-1][1], force)
    np.testing.assert_array_equal(display.image_artist.get_array()[0, 0], [255, 0, 0])
    # Deadbands affect the diagram, never the recorded curves or signed values.
    np.testing.assert_array_equal(display.force_position, [0.0, -2.0, 0.0])
    assert display.force_z.radius == pytest.approx(0.4)
    assert "-5.00" in display.force_values[2].get_text()
    assert len(display.force_axes[2].lines[0].get_ydata()) == 1
    assert display.force_axes[2].lines[0].get_ydata()[0] == -5.0


def test_normal_force_uses_color_instead_of_radius(display, monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    display.update_force([2.0, -1.0, -1.0])
    low_color = display.force_z.get_facecolor()
    radius = display.force_z.radius
    assert low_color[2] > low_color[0]  # Small magnitudes are blue.

    monkeypatch.setattr(module.time, "monotonic", lambda: 0.2)
    display.update_force([2.0, -1.0, -4.0])
    high_color = display.force_z.get_facecolor()
    assert high_color[0] > high_color[2]  # Large magnitudes are red.
    assert display.force_z.radius == radius
    np.testing.assert_array_equal(display.force_position, [2.0, -1.0, 0.0])

    monkeypatch.setattr(module.time, "monotonic", lambda: 0.4)
    display.update_force([2.0, -1.0, 4.0])
    np.testing.assert_allclose(display.force_z.get_facecolor(), high_color)
    assert display.force_values[2].get_text() == "+4.00 N"
    np.testing.assert_array_equal(display.force_lines[2].get_ydata(), [-1.0, -4.0, 4.0])


def test_normal_colorbar_expands_without_shrinking_or_losing_force(display, monkeypatch):
    assert display.normal_colorbar.mappable.norm.vmin == 0.0
    assert display.normal_colorbar.mappable.norm.vmax == 5.0
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    display.update_force([0.0, 0.0, -12.0])
    assert display.normal_colorbar.mappable.norm.vmax >= 12.0
    upper = display.normal_colorbar.mappable.norm.vmax
    assert display.force_values[2].get_text() == "-12.00 N"
    np.testing.assert_allclose(
        display.force_z.get_facecolor(), display.normal_colorbar.mappable.to_rgba(12.0),
    )

    monkeypatch.setattr(module.time, "monotonic", lambda: 0.2)
    display.update_force([0.0, 0.0, -1.0])
    assert display.normal_colorbar.mappable.norm.vmax == upper


@pytest.mark.parametrize("normal, visible", [(0.0, False), (-0.49, False), (-0.5, True)])
def test_normal_color_marker_respects_deadband(display, normal, visible):
    display.update_force([0.0, 0.0, normal])
    assert display.force_z.get_visible() == visible
    assert display.force_history[-1][1][2] == normal


def test_rolling_history_and_close(display, monkeypatch):
    for elapsed in (0.0, 5.0, 30.0, 35.0):
        monkeypatch.setattr(module.time, "monotonic", lambda: elapsed)
        display.update_force([0.0, 0.0, -2.0])
    assert [sample[0] for sample in display.force_history] == [5.0, 30.0, 35.0]
    assert display.force_axes[-1].get_xlim() == (5.0, 35.0)
    assert display.vis.poll_events()
    display.close()
    display.close()
    assert not display.vis.poll_events()
    module.cv2.destroyWindow.assert_called_once()


def test_redraw_throttle_keeps_samples(display, monkeypatch):
    for elapsed in (0.0, 0.01, 0.02):
        monkeypatch.setattr(module.time, "monotonic", lambda: elapsed)
        display.update_force([0.0, 0.0, -elapsed])
    assert len(display.force_history) == 3
    assert module.cv2.imshow.call_count == 1


@pytest.mark.parametrize("force", ([1, 2], [0, 0, np.nan], [np.inf, 0, 0]))
def test_invalid_force_is_rejected(display, force):
    with pytest.raises(ValueError, match="three finite"):
        display.update_force(force)


def test_closed_window_stops_loop(display, monkeypatch):
    monkeypatch.setattr(module.cv2, "getWindowProperty", lambda *args: 0)
    assert not display.vis.poll_events()


def test_split_diagram_does_not_accumulate_or_change_signed_history(monkeypatch):
    monkeypatch.setattr(module.open3d.visualization, "Visualizer", Mock())
    for name in ("namedWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(module.cv2, name, Mock())
    monkeypatch.setattr(module.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    split = module.Visualizer((0.5, 0.5, 0.5))
    try:
        split.update_force([2, -3, -5])
        split.update_force([2, -3, -5])
        np.testing.assert_array_equal(split.force_position, [2, -3, 0])
        vertices = np.asarray(split.force_z.vertices)
        assert np.linalg.norm(vertices[1] - split.force_position) == pytest.approx(0.6)
        split.update_force([0.1, -0.2, -0.3])
        np.testing.assert_array_equal(split.force_position, [0, 0, 0])
        assert not split.visible[2]
        np.testing.assert_array_equal(split.force_history[-1][1], [0.1, -0.2, -0.3])
    finally:
        split.close()
