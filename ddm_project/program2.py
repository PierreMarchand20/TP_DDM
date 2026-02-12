import gmsh
import skfem
import numpy as np 
from skfem.helpers import dot, grad  
from skfem.visuals.matplotlib import plot
from matplotlib import pyplot as plt


from ddm_playground.mesh.gmsh import GmshContextManager, GmshOptions
from ddm_playground.mesh.plot import plot_mesh, plot_submesh
from ddm_playground.mesh.overlap import add_overlap 

dim: int = 1
nb_partition: int = 2
gmsh_options = GmshOptions(mesh_name="mesh")

with GmshContextManager(gmsh_options) as mesh_generator:
    
    lc = 0.1  # characteristic length (mesh size)

    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    gmsh.model.geo.synchronize()
    mesh_global = mesh_generator.generate(dim, nb_partition)


submeshes, neighbors, intersections, partition_of_unity, _ = add_overlap(
        mesh_global, 1
    )


# matplotlib visualization
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_title(f"{dim}D Mesh from GMSH")
ax1.axis("equal")
ax1.set_xlim(0,1)
plot_mesh(ax1, submeshes[0])
ax2 = fig.add_subplot(122)
ax2.set_title(f"{dim}D Mesh from GMSH")
ax2.axis("equal")
ax2.set_xlim(0,1)
plot_mesh(ax2, submeshes[1])
plt.show()

