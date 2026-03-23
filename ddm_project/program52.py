import gmsh
import skfem
import numpy as np 
from skfem.helpers import dot, grad  
from skfem.visuals.matplotlib import plot
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from ddm_playground.mesh.gmsh import GmshContextManager, GmshOptions
from ddm_playground.mesh.plot import plot_mesh, plot_submesh
from ddm_playground.mesh.overlap import add_overlap 
from program5 import global_matrix_vector_product

dim: int = 1
nb_partition: int = 2
gmsh_options = GmshOptions(mesh_name="mesh")

with GmshContextManager(gmsh_options) as mesh_generator:
    
    lc = 0.05  # characteristic length (mesh size)

    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    gmsh.model.geo.synchronize()

    boundaries = gmsh.model.getEntities(dim=dim - 1)
    for i, (_, boundary_tag) in enumerate(boundaries):
        gmsh.model.addPhysicalGroup(dim - 1, [boundary_tag], name=f"boundary_{i}")
    mesh_global = mesh_generator.generate(dim, nb_partition)


# Partition meshes
submeshes, neighbors, intersections, partition_of_unity, ovr_subdomain_to_global = add_overlap(
        mesh_global, 1
    )

# Create local FEM meshes and basis
sk_submeshes = []
Vhs = []
for i in range(0,nb_partition):
    nodes = submeshes[i].nodes.T
    elements = submeshes[i].elements.T

    if dim == 1:
        nodes =nodes[0,:]
        skmesh=skfem.MeshLine(nodes,elements)
    elif dim == 2:
        nodes = nodes[0:2,:]
        skmesh =skfem.MeshTri(nodes,elements)
    sk_submeshes.append(skmesh)
    if dim == 1:
        Vhs.append(skfem.Basis(skmesh, skfem.ElementLineP1()))
    elif dim == 2:
        Vhs.append(skfem.Basis(skmesh, skfem.ElementTriP1()))


nodes = mesh_global.nodes.T
elements =mesh_global.elements.T


if dim == 1:
    skmesh_global =skfem.MeshLine(nodes[0,:],elements)
    Vh_global=skfem.Basis(skmesh, skfem.ElementLineP1())
elif dim == 2:
    skmesh_global =skfem.MeshTri(nodes[0:2,:],elements)
    Vh_global=skfem.Basis(skmesh, skfem.ElementTriP1())

dofs_global = nodes.shape[1]

# Idendify mapping from interfaces
exchange_indices = [] # exchange_indices[i][n] gives the indices of the overlap of submesh i with submesh n in the local numbering of submesh n. It corresponds with the overlap in  intersections[i] corresponding to neighbour n in neighbours[i]
overlap_nodes =[] # local indices of all overlap nodes
ext_boundary_nodes = [] # local indices for exterior boundary
boundary_nodes = [] # local indices for boundary nodes and overlap nodes

# Boundary tags for this problem
if dim == 1:
    ext_boundary_tags = [("boundary_0", 0, 1),("boundary_1", 0, 2)]
elif dim == 2:
    ext_boundary_tags = [("boundary_0", 1, 1),("boundary_1", 1, 2)]
    

# Create mapping
for i in range(0,nb_partition):
    # Obtain overlapping nodes between subdomains
    interface_elem = submeshes[i].physical_group_elements[('interface', dim-1, None)]
    nodes_on_interface = np.unique(np.concatenate(interface_elem))

    # Obtain exterior boundary elements for boundary conditions
    nodes_on_exterior_boundary= []
    for ext_boundary_tag in ext_boundary_tags:
        if ext_boundary_tag in ( submeshes[i].physical_group_elements).keys():
            exterior_boundary_elem = submeshes[i].physical_group_elements[ext_boundary_tag]
            nodes_on_exterior_boundary.append(np.unique(np.concatenate(exterior_boundary_elem)))

    # Store relevant boundary data
    overlap_nodes.append(nodes_on_interface)
    ext_boundary_nodes.append(nodes_on_exterior_boundary)
    if len(nodes_on_exterior_boundary) >0 :
        boundary_nodes.append(np.hstack((nodes_on_interface,*tuple(nodes_on_exterior_boundary))))
    else:
        boundary_nodes.append(nodes_on_interface)

    # Obtain neighbours and overlaps
    neigh = neighbors[i]
    inters = intersections[i]
    args={}
    for j, n in enumerate(neigh):
        # Obtain neighbours of neighbour n
        other_neigh = neighbors[n]
        for jj,nn in enumerate(other_neigh):
            if nn == i:
                # Extract the index numbering of the overlap in the neighbour numbering
                temp= intersections[n][jj]
        # Store variables
        args[n] = temp
    exchange_indices.append(args)

print(f"neighbours={neighbors}")
print(f"overlap_nodes={overlap_nodes}")
print(f"exchange_indices={exchange_indices}")
print(f"boundary_nodes = {boundary_nodes}" )
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
xs_init = []
initial_interface_conditions = []

for i in range(0,nb_partition):
    A = a.assemble(Vhs[i])    
    b = l.assemble(Vhs[i])
    As.append(A)
    bs.append(b)
    local_ext_boundary_nodes = boundary_nodes[i]
    Linear_system = skfem.enforce(As[i], bs[i],D=local_ext_boundary_nodes)
    xs.append([])


# Specify number of iterations
nb_iterations= 5



# y = global_matrix_vector_product(As,xs,partition_of_unity,neighbors,intersections,exchange_indices,ovr_subdomain_to_global,dofs_global)
print(f"y = {y}")

