import argparse
from itertools import cycle

import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from ddm_playground.mesh.gmsh import GmshContextManager, GmshOptions
from ddm_playground.mesh.overlap import add_overlap
from ddm_playground.mesh.plot import plot_mesh, plot_submesh
from matplotlib import cm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create domain decomposition with overlap."
    )

    # Dimension of the problem
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Problem dimension (1D or 2D or 3D)",
    )

    # Number of partitions (for domain decomposition)
    parser.add_argument(
        "--nparts", type=int, default=4, help="Number of partitions (e.g. for ASM/RAS)"
    )

    # Number of additional layers of overlap (for domain decomposition)
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Number of addition layers of overlap (e.g. for ASM/RAS)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dim = args.dim
    nb_partitions = args.nparts
    additional_overlap = args.overlap
    gmsh_options = GmshOptions(mesh_name="mesh")

    with GmshContextManager(gmsh_options) as mesh_generator:
        lc = 0.1  # characteristic length (mesh size)

        if dim == 1:
            p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
            p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
            l1 = gmsh.model.geo.addLine(p1, p2)
            gmsh.model.geo.synchronize()

        elif dim == 2:
            p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
            p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
            p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
            p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)

            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)
            cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.synchronize()

        elif dim == 3:
            cube = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

        # Physical names for boundary
        boundaries = gmsh.model.getEntities(dim=dim - 1)
        for i, (_, boundary_tag) in enumerate(boundaries):
            gmsh.model.addPhysicalGroup(dim - 1, [boundary_tag], name=f"boundary_{i}")

        mesh = mesh_generator.generate(dim, nb_partitions)

    submeshes, neighbors, intersections, partition_of_unity, _ = add_overlap(
        mesh, additional_overlap
    )

    # matplotlib visualization
    ncols = 2 if nb_partitions > 1 else 1
    nrow = 2 if nb_partitions > 2 else 1
    fig = plt.figure()
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    z = mesh.nodes[:, 2]

    colors = plt.cm.tab10.colors

    for partition_index in range(min(nb_partitions, 4)):
        color_iter = cycle(colors)
        ax = (
            fig.add_subplot(nrow, ncols, partition_index + 1, projection="3d")
            if dim == 3
            else fig.add_subplot(nrow, ncols, partition_index + 1)
        )
        plot_mesh(ax, mesh)

        # Partition
        plot_submesh(
            ax,
            mesh,
            dim,
            mesh.partitions_elements[partition_index],
            color=next(color_iter),
            label=f"Partition {partition_index}",
            # linestyle="none",
            markersize=5,
        )

        # Subdomain with overlap
        plot_mesh(
            ax,
            submeshes[partition_index],
            linestyle="none",
            label=f"Nodes in subdomain {partition_index}",
            color=next(color_iter),
            marker="x",
            markersize=10,
        )

        # Physical boundary of subdomain
        for (
            boundary_name,
            boundary_dim,
            _unused,
        ), elements in submeshes[partition_index].physical_group_elements.items():
            plot_submesh(
                ax,
                submeshes[partition_index],
                boundary_dim,
                elements,
                label=boundary_name,
                color=next(color_iter),
            )
        ax.legend()
        ax.set_title(f"Subdomain {partition_index}")
        ax.axis("equal")

    if dim < 3:
        fig = plt.figure()
        for partition_index in range(min(nb_partitions, 4)):
            color_iter = cycle(colors)
            ax = fig.add_subplot(nrow, ncols, partition_index + 1)

            if dim == 1:
                # Partition
                plot_submesh(
                    ax,
                    mesh,
                    dim,
                    mesh.partitions_elements[partition_index],
                    color=next(color_iter),
                    label=f"Partition {partition_index}",
                    # linestyle="none",
                    markersize=5,
                )

                # Subdomain with overlap
                plot_mesh(
                    ax,
                    submeshes[partition_index],
                    linestyle="none",
                    label=f"Nodes in subdomain {partition_index}",
                    color=next(color_iter),
                    marker="x",
                    markersize=10,
                )
                local_x = submeshes[partition_index].nodes[:, 0]
                perm = np.argsort(local_x)
                print(partition_index, local_x, partition_of_unity[partition_index])
                ax.plot(
                    local_x[perm],
                    partition_of_unity[partition_index][perm],
                    marker="o",
                    color=next(color_iter),
                    label="Partition of unity",
                )
                ax.legend()

            elif dim == 2:
                norm = plt.Normalize(vmin=0, vmax=1)
                cmap = plt.get_cmap("viridis", nb_partitions)
                local_x = submeshes[partition_index].nodes[:, 0]
                local_y = submeshes[partition_index].nodes[:, 1]
                triang = mtri.Triangulation(
                    local_x, local_y, submeshes[partition_index].elements
                )
                ax.tripcolor(
                    triang, partition_of_unity[partition_index], shading="gouraud"
                )
                plt.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax,
                    label="Partition of unity",
                )

            plot_mesh(ax, mesh)
            ax.set_title(f"Subdomain {partition_index}")
    plt.show()


if __name__ == "__main__":
    main()
