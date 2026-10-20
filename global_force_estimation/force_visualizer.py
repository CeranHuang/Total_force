import numpy as np
import open3d


class Visualizer:
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name="Global Force (N)", width=1200, height=900)
        self.force_origin = [0, 0, 0]

        self.force_x = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.05, cone_radius=0.12, cylinder_height=1, cone_height=0.2
        )
        self.force_y = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.05, cone_radius=0.12, cylinder_height=1, cone_height=0.2
        )
        self.force_z = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.05, cone_radius=0.12, cylinder_height=1, cone_height=0.2
        )

        self.force_x.rotate(
            open3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]), [0, 0, 0]
        )
        self.force_y.rotate(
            open3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0]), [0, 0, 0]
        )
        self.force_x.paint_uniform_color([1, 0, 0])
        self.force_y.paint_uniform_color([0, 1, 0])
        self.force_z.paint_uniform_color([0, 0, 1])

        self.force_arrows = (self.force_x, self.force_y, self.force_z)
        self.reverse_rotations = (
            open3d.geometry.get_rotation_matrix_from_xyz([0, np.pi, 0]),
            open3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]),
            open3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]),
        )
        self.last_force = np.ones(3)
        for arrow in self.force_arrows:
            arrow.translate(self.force_origin)
            self.vis.add_geometry(arrow)

        self.ctr = self.vis.get_view_control()
        self.ctr.set_zoom(0.6)
        self.ctr.set_lookat([0, 0, 0])
        self.vis.update_renderer()

    def update_force(self, force):
        force = np.asarray(force, dtype=np.float64)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force must contain three finite values (Fx, Fy, Fz).")

        for arrow, previous, current, reverse in zip(
            self.force_arrows, self.last_force, force, self.reverse_rotations
        ):
            if (current < 0) != (previous < 0):
                arrow.rotate(reverse, center=self.force_origin)
            scale = max(abs(current), 1e-6) / max(abs(previous), 1e-6)
            arrow.scale(float(scale), center=self.force_origin)
            self.vis.update_geometry(arrow)
        self.last_force = force.copy()