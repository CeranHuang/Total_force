from collections import deque
import time

import cv2
import numpy as np
import open3d
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class Visualizer:
    def __init__(self, force_threshold_n=(1, 1, 1)):
        self.force_threshold_n = np.asarray(force_threshold_n, dtype=np.float64)
        if (self.force_threshold_n.shape != (3,)
                or not np.isfinite(self.force_threshold_n).all()
                or (self.force_threshold_n < 0).any()):
            raise ValueError("force_threshold_n must contain three finite non-negative values (Fx, Fy, Fz).")
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name="Global Force (N)", width=1200, height=900)
        self.force_origin = [0, 0, 0]

        self.force_position = np.asarray(self.force_origin, dtype=np.float64)
        angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        self.force_z = open3d.geometry.TriangleMesh()
        self.force_z.vertices = open3d.utility.Vector3dVector(np.vstack([
            [0, 0, 0],
            np.column_stack([0.12 * np.cos(angles), 0.12 * np.sin(angles),
                             np.zeros_like(angles)]),
        ]))
        self.force_z.triangles = open3d.utility.Vector3iVector([
            [0, index + 1, (index + 1) % len(angles) + 1]
            for index in range(len(angles))
        ])

        self.force_z.paint_uniform_color([0, 0, 1])

        self.last_force = np.ones(3)
        self.visible = np.ones(3, dtype=bool)
        self.force_z.translate(self.force_origin)
        self.vis.add_geometry(self.force_z)
        self.vis.get_render_option().mesh_show_back_face = True

        self.ctr = self.vis.get_view_control()
        self.view_half_extent = 0.0
        self._fit_view(2.0)
        self.ctr.set_lookat([0, 0, 0])
        self.ctr.set_front([0, 0, 1])
        self.ctr.set_up([0, 1, 0])
        self.vis.remove_geometry(self.force_z, reset_bounding_box=False)
        self.visible[2] = False
        self.vis.update_renderer()

        self.plot_window = "Force history (N)"
        self.force_history = deque(maxlen=10000)
        self.start_time = None
        self.last_plot_time = None
        self.plot_closed = False
        self.force_figure = Figure(figsize=(8, 7), dpi=100)
        self.force_canvas = FigureCanvasAgg(self.force_figure)
        self.force_axes = self.force_figure.subplots(3, 1, sharex=True)
        self.force_lines = []
        for axis, name, color in zip(self.force_axes, ("Fx", "Fy", "Fz"), ("red", "green", "blue")):
            line, = axis.plot([], [], color=color, linewidth=1.5)
            self.force_lines.append(line)
            axis.set_ylabel(f"{name} (N)")
            axis.grid(True, alpha=0.3)
            axis.set_xlim(0, 30)
        self.force_axes[0].set_title("Force vs time")
        self.force_axes[-1].set_xlabel("Time (s)")
        self.force_figure.tight_layout()
        cv2.namedWindow(self.plot_window, cv2.WINDOW_NORMAL)

    def _update_force_plot(self, force):
        if self.plot_closed:
            return
        if cv2.getWindowProperty(self.plot_window, cv2.WND_PROP_VISIBLE) < 1:
            self.plot_closed = True
            return
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        self.force_history.append((elapsed, force.copy()))
        while self.force_history[0][0] < elapsed - 30:
            self.force_history.popleft()
        # Keep every sample, but limit expensive plot redraws to 10 Hz.
        if self.last_plot_time is not None and now - self.last_plot_time < 0.1:
            return
        times, forces = zip(*self.force_history)
        forces = np.asarray(forces)
        for index, (axis, line) in enumerate(zip(self.force_axes, self.force_lines)):
            line.set_data(times, forces[:, index])
            axis.relim()
            axis.autoscale_view(scalex=False)
        self.force_axes[-1].set_xlim(max(0, elapsed - 30), max(30, elapsed))
        self.force_canvas.draw()
        rgba = np.asarray(self.force_canvas.buffer_rgba())
        cv2.imshow(self.plot_window, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
        self.last_plot_time = now

    def close(self):
        self.vis.destroy_window()
        if not self.plot_closed:
            cv2.destroyWindow(self.plot_window)
            self.plot_closed = True

    def _fit_view(self, required_extent):
        if required_extent <= self.view_half_extent:
            return
        self.view_half_extent = max(2.0, required_extent, self.view_half_extent * 2)
        # Set camera bounds independently of the circle's initial tiny radius.
        bounds = open3d.geometry.PointCloud()
        bounds.points = open3d.utility.Vector3dVector(
            np.asarray(self.force_origin) + np.array([-1, 1])[:, None] * self.view_half_extent
        )
        self.vis.add_geometry(bounds, reset_bounding_box=True)
        self.vis.remove_geometry(bounds, reset_bounding_box=False)
        self.ctr.set_zoom(0.7)

    def update_force(self, force):
        force = np.asarray(force, dtype=np.float64)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force must contain three finite values (Fx, Fy, Fz).")
        self._update_force_plot(force)

        # Tangential force is a position in the XY plane, never accumulated motion.
        tangent = np.where(np.abs(force[:2]) >= self.force_threshold_n[:2], force[:2], 0)
        position = np.asarray(self.force_origin) + [tangent[0], tangent[1], 0]
        self.force_z.translate(position - self.force_position)
        self.force_position = position
        radius = 0.12 * abs(force[2])
        self._fit_view(1.25 * (np.max(np.abs(tangent)) + radius))

        # Keep the planar circle facing the camera even after mouse interaction.
        self.ctr.set_front([0, 0, 1])
        self.ctr.set_up([0, 1, 0])
        self.ctr.set_lookat(self.force_origin)
        current = force[2]
        if abs(current) < self.force_threshold_n[2] or current == 0:
            if self.visible[2]:
                self.vis.remove_geometry(self.force_z, reset_bounding_box=False)
                self.visible[2] = False
            return

        scale = abs(current) / abs(self.last_force[2])
        self.force_z.scale(float(scale), center=self.force_position)
        if not self.visible[2]:
            self.vis.add_geometry(self.force_z, reset_bounding_box=False)
            self.visible[2] = True
        self.vis.update_geometry(self.force_z)
        self.last_force[2] = current
