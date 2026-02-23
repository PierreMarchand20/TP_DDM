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

sk_submeshes = []
Vhs = []
for i in range(0,nb_partition):
    nodes = submeshes[i].nodes.T
    elements = submeshes[i].elements.T

    nodes =nodes[0,:]
    skmesh=skfem.MeshLine(nodes,elements)
    sk_submeshes.append(skmesh)
    Vhs.append(skfem.Basis(skmesh, skfem.ElementLineP1()))

# Define bilinear and linear form
@skfem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))

@skfem.LinearForm
def l(v, _):
    f = 1
    return f * v

# Assemble local problem
As= []
bs=[]
Ds=[]
xs = []

for i in range(0,nb_partition):
    A = a.assemble(Vhs[i])
    b = l.assemble(Vhs[i])
    As.append(A)
    bs.append(b)
    boundary_elem = submeshes[i].physical_group_elements[('interface', 0, None)]
    D = Vhs[i].get_dofs(boundary_elem)
    Ds.append(D)
    x = skfem.solve(*skfem.condense(A, b, D=D))
    xs.append(x)

for i in range(0,nb_partition):
    boundary_elem = submeshes[i].physical_group_elements[('interface', 0, None)]
    neigh = neighbors[i]
    inter = intersections[i]
