from collections import deque
import time

import cv2
import numpy as np
import open3d
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch


# Axis colors follow the orange / green / blue convention in BioTacTip Figs. 8 and 10.
FORCE_COLORS = ("#D99427", "#389A78", "#4878CF")
INK = "#24364B"
MUTED = "#728095"


def _style_force_axes(axes):
    for axis, name, color in zip(axes, ("x", "y", "z"), FORCE_COLORS):
        axis.set_facecolor("#F7F9FC")
        axis.set_ylabel(f"$F_{name}$ (N)", color=color, fontsize=11)
        axis.tick_params(colors=MUTED, labelsize=9, length=0, pad=7)
        axis.grid(True, color="#DFE5ED", linewidth=0.7, linestyle="--")
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#D8E0EA")
        axis.axhline(0, color="#B3BFCE", linewidth=0.8, zorder=1)


class Visualizer:
    def __init__(self, force_threshold_n=(1, 1, 1)):
        self.force_threshold_n = np.asarray(force_threshold_n, dtype=np.float64)
        if (self.force_threshold_n.shape != (3,)
                or not np.isfinite(self.force_threshold_n).all()
                or (self.force_threshold_n < 0).any()):
            raise ValueError("force_threshold_n must contain three finite non-negative values (Fx, Fy, Fz).")
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name="Global Force (N)", width=1200, height=900)
        self.vis.get_render_option().background_color = np.array([0.965, 0.976, 0.988])
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

        self.force_z.paint_uniform_color([0.282, 0.471, 0.812])

        self.last_force = np.ones(3)
        self.visible = np.ones(3, dtype=bool)
        self.force_z.translate(self.force_origin)
        self.vis.add_geometry(self.force_z)
        # Reference rings and crosshairs give the moving disk a visible origin and scale.
        points, lines = [], []
        for ring_radius in (0.5, 1.0, 2.0):
            start = len(points)
            points.extend(np.column_stack([
                ring_radius * np.cos(angles), ring_radius * np.sin(angles),
                np.full_like(angles, -0.01),
            ]))
            lines.extend([[start + i, start + (i + 1) % len(angles)] for i in range(len(angles))])
        start = len(points)
        points.extend([[-2, 0, -0.01], [2, 0, -0.01], [0, -2, -0.01], [0, 2, -0.01]])
        lines.extend([[start, start + 1], [start + 2, start + 3]])
        self.reference_grid = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines),
        )
        self.reference_grid.paint_uniform_color([0.78, 0.82, 0.88])
        self.vis.add_geometry(self.reference_grid, reset_bounding_box=False)
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
        self.force_values = []
        for axis, name, color in zip(self.force_axes, ("Fx", "Fy", "Fz"), FORCE_COLORS):
            line, = axis.plot([], [], color=color, linewidth=2)
            self.force_lines.append(line)
            axis.set_xlim(0, 30)
            self.force_values.append(axis.text(
                0.98, 0.9, "0.00 N", transform=axis.transAxes,
                ha="right", va="top", color=color, fontsize=12, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
            ))
        _style_force_axes(self.force_axes)
        self.force_figure.suptitle("GLOBAL FORCE  /  LIVE PREDICTION", color=INK,
                                   fontsize=14, fontweight="bold", x=0.12, ha="left")
        self.force_axes[-1].set_xlabel("Time (s)")
        self.force_figure.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=1.5)
        self.force_figure.text(0.12, 0.02, "Signed forces  |  Compression: Fz < 0  |  Last 30 s",
                               fontsize=9, color=MUTED)
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
            self.force_values[index].set_text(f"{force[index]:+.2f} N")
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


class DashboardVisualizer:
    """Paper-style camera, force-space diagram and signed time histories."""

    def __init__(self, force_threshold_n=(1, 1, 1)):
        self.force_threshold_n = np.asarray(force_threshold_n, dtype=np.float64)
        if (self.force_threshold_n.shape != (3,)
                or not np.isfinite(self.force_threshold_n).all()
                or (self.force_threshold_n < 0).any()):
            raise ValueError("force_threshold_n must contain three finite non-negative values (Fx, Fy, Fz).")
        self.plot_window = "Global force | BioTacTip-inspired dashboard"
        self.plot_closed = False
        self.vis = self  # Preserve the existing entry point's vis.poll_events() interface.
        self.force_history = deque(maxlen=10000)
        self.start_time = None
        self.last_plot_time = None
        self.force_position = np.zeros(3)
        self.view_half_extent = 5.0
        self.force_figure = Figure(figsize=(14.4, 9), dpi=100, facecolor="white")
        self.force_canvas = FigureCanvasAgg(self.force_figure)
        fig = self.force_figure
        fig.text(0.045, 0.943, "GLOBAL FORCE ESTIMATION", fontsize=21,
                 fontweight="bold", color=INK)
        fig.text(0.045, 0.91, "TACTILE SENSING   /   Camera + force-space view + time histories",
                 fontsize=10, color=MUTED)
        fig.text(0.955, 0.943, "PREDICTION", ha="right", fontsize=10, color=FORCE_COLORS[1])
        self.force_values = []
        for x, name, color in zip((0.045, 0.285, 0.525), ("x", "y", "z"), FORCE_COLORS):
            fig.text(x, 0.86, f"$F_{name}$  /  " + ("NORMAL" if name == "z" else "TANGENTIAL"),
                     fontsize=10, color=color)
            self.force_values.append(fig.text(x, 0.812, "+0.00 N", fontsize=27,
                                               fontweight="bold", color=color))
        self.magnitude_text = fig.text(0.78, 0.827, "", fontsize=18, fontweight="bold",
                                       fontfamily="Microsoft YaHei", color=INK)

        fig.text(0.045, 0.765, "(A)  Raw image", fontsize=11, color=INK, fontweight="bold")
        self.image_axis = fig.add_axes([0.045, 0.455, 0.33, 0.285])
        self.image_axis.set_axis_off()
        self.image_artist = self.image_axis.imshow(np.full((240, 320, 3), 237, dtype=np.uint8))
        self.image_placeholder = self.image_axis.text(
            0.5, 0.5, "Waiting for camera frame", transform=self.image_axis.transAxes,
            ha="center", color=MUTED,
        )

        fig.text(0.045, 0.408, "(B)  FORCE-SPACE VIEW", fontsize=11, color=INK, fontweight="bold")
        self.tangent_axis = fig.add_axes([0.075, 0.115, 0.24, 0.26], aspect="equal")
        axis = self.tangent_axis
        axis.set_facecolor("#F7F9FC")
        axis.grid(True, color="#DFE5ED", linewidth=0.7, linestyle="--")
        axis.set_axisbelow(True)
        axis.axhline(0, color="#B3BFCE", linewidth=0.9)
        axis.axvline(0, color="#B3BFCE", linewidth=0.9)
        axis.set_xlabel("$F_x$ (N)", color=FORCE_COLORS[0], labelpad=3)
        axis.set_ylabel("$F_y$ (N)", color=FORCE_COLORS[1], labelpad=3)
        axis.tick_params(colors=MUTED, labelsize=9, length=0)
        for spine in axis.spines.values():
            spine.set_color("#D8E0EA")
        normal_colors = ScalarMappable(norm=Normalize(vmin=0, vmax=5), cmap="jet")
        colorbar_axis = fig.add_axes([0.315, 0.125, 0.012, 0.24])
        self.normal_colorbar = fig.colorbar(normal_colors, cax=colorbar_axis)
        colorbar_axis.set_title("$|F_z|$ (N)", fontsize=10, color=INK, pad=9)
        colorbar_axis.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
        self.normal_colorbar.outline.set_edgecolor("#D8E0EA")
        # Marker size is constant; only color encodes normal-force magnitude.
        self.force_z = Circle((0, 0), 0.4, facecolor=normal_colors.to_rgba(0), edgecolor=INK,
                              linewidth=1, zorder=3, visible=False)
        axis.add_patch(self.force_z)
        self.tangent_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                             mutation_scale=16, color=INK, linewidth=1.5, zorder=4)
        axis.add_patch(self.tangent_arrow)
        axis.plot([0], [0], marker="+", color=INK, markersize=9, zorder=5)
        fig.text(0.045, 0.06, "Centre = (Fx, Fy); colour = |Fz| (N)", fontsize=9, color=MUTED)
        fig.text(0.045, 0.039, "Force encoding only; not contact location", fontsize=9, color=MUTED)

        fig.text(0.465, 0.765, "(C)  SIGNED FORCE HISTORIES", fontsize=11, color=INK, fontweight="bold")
        self.force_axes, self.force_lines = [], []
        for y, color in zip((0.585, 0.365, 0.145), FORCE_COLORS):
            axis = fig.add_axes([0.465, y, 0.49, 0.155])
            line, = axis.plot([], [], color=color, linewidth=2)
            self.force_axes.append(axis)
            self.force_lines.append(line)
            axis.set_xlim(0, 30)
            axis.set_ylim(-5, 5)
        _style_force_axes(self.force_axes)
        for axis in self.force_axes[:-1]:
            axis.tick_params(labelbottom=False)
        self.force_axes[-1].set_xlabel("Time (s)", color=MUTED, labelpad=7)
        self.elapsed_text = fig.text(0.955, 0.765, "Last 30 s", ha="right", fontsize=9, color=MUTED)
        fig.text(0.465, 0.053, "Compression: Fz < 0   /   Curves and values retain all predictions",
                 fontsize=9, color=MUTED)
        thresholds = ", ".join(f"{value:g}" for value in self.force_threshold_n)
        fig.text(0.465, 0.03, f"Diagram deadband [Fx, Fy, Fz]: [{thresholds}] N     |     Q to exit",
                 fontsize=9, color=MUTED)
        cv2.namedWindow(self.plot_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.plot_window, 1440, 900)

    def poll_events(self):
        if self.plot_closed:
            return False
        try:
            visible = cv2.getWindowProperty(self.plot_window, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if not visible:
            self.plot_closed = True
        return visible

    def update_force(self, force, image=None):
        force = np.asarray(force, dtype=np.float64)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force must contain three finite values (Fx, Fy, Fz).")
        if not self.poll_events():
            return
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        self.force_history.append((elapsed, force.copy()))
        while self.force_history[0][0] < elapsed - 30:
            self.force_history.popleft()
        # Record every prediction, but keep the existing 10 Hz drawing budget.
        if self.last_plot_time is not None and now - self.last_plot_time < 0.1:
            return
        if image is not None:
            height, width = image.shape[:2]
            thumbnail = cv2.resize(image, (480, max(1, round(480 * height / width))))
            self.image_artist.set_data(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
            self.image_artist.set_extent((0, width, height, 0))
            self.image_placeholder.set_visible(False)
        for text, value in zip(self.force_values, force):
            text.set_text(f"{value:+.2f} N")
        self.magnitude_text.set_text(f"Total Force |F|  {np.linalg.norm(force):.2f} N")
        tangent = np.where(np.abs(force[:2]) >= self.force_threshold_n[:2], force[:2], 0)
        self.force_position = np.array([*tangent, 0.0])
        self.force_z.center = tuple(tangent)
        normal_magnitude = abs(force[2])
        normal_colors = self.normal_colorbar.mappable
        if normal_magnitude > normal_colors.norm.vmax:
            # Grow in 5 N steps and retain the scale when the force decreases.
            normal_colors.set_clim(0, 5 * np.ceil(normal_magnitude / 5))
        self.force_z.set_facecolor(normal_colors.to_rgba(normal_magnitude))
        self.force_z.set_visible(abs(force[2]) >= self.force_threshold_n[2] and force[2] != 0)
        self.tangent_arrow.set_positions((0, 0), tuple(tangent))
        self.tangent_arrow.set_visible(bool(np.any(tangent)))
        needed = 1.2 * (np.max(np.abs(tangent)) + self.force_z.radius)
        if needed > self.view_half_extent:
            self.view_half_extent = max(needed, 1.5 * self.view_half_extent)
        self.tangent_axis.set_xlim(-self.view_half_extent, self.view_half_extent)
        self.tangent_axis.set_ylim(-self.view_half_extent, self.view_half_extent)

        times, forces = zip(*self.force_history)
        forces = np.asarray(forces)
        for index, (axis, line) in enumerate(zip(self.force_axes, self.force_lines)):
            line.set_data(times, forces[:, index])
            # Expand only when needed, so steady readings do not make axes jitter.
            low, high = axis.get_ylim()
            axis.set_ylim(min(low, float(forces[:, index].min()) - 0.5),
                          max(high, float(forces[:, index].max()) + 0.5))
            axis.set_xlim(max(0, elapsed - 30), max(30, elapsed))
        self.elapsed_text.set_text(f"Last 30 s   /   t = {elapsed:.1f} s")
        self.force_canvas.draw()
        rgba = np.asarray(self.force_canvas.buffer_rgba())
        cv2.imshow(self.plot_window, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
        self.last_plot_time = now

    def close(self):
        if not self.plot_closed:
            cv2.destroyWindow(self.plot_window)
            self.plot_closed = True
