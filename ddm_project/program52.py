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
from program5 import *
import scipy
from tools import *
from program3 import *
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
    skmesh_global =skfem.MeshLine(nodes[0:1,:],elements)
    Vh_global=skfem.Basis(skmesh_global, skfem.ElementLineP1())
elif dim == 2:
    skmesh_global =skfem.MeshTri(nodes[0:2,:],elements)
    Vh_global=skfem.Basis(skmesh_global, skfem.ElementTriP1())

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

    ext_boundary_nodes.append(np.hstack(*tuple(nodes_on_exterior_boundary)))
    # Store relevant boundary data
    overlap_nodes.append(nodes_on_interface)
    # ext_boundary_nodes.append(nodes_on_exterior_boundary)
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
print(f"intersection = {intersections}")

# Global boundary nodes
mesh_global
print(mesh_global.physical_group_elements)
nodes_on_exterior_boundary = []
for ext_boundary_tag in ext_boundary_tags:
    if ext_boundary_tag in ( mesh_global.physical_group_elements).keys():
        exterior_boundary_elem =mesh_global.physical_group_elements[ext_boundary_tag]
        nodes_on_exterior_boundary.append(np.unique(np.concatenate(exterior_boundary_elem)))
boundary_nodes_global = np.concatenate(nodes_on_exterior_boundary)
print(f'boundary_nodes_global={boundary_nodes_global}')

# Define bilinear and linear form

@skfem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))

@skfem.LinearForm
def l(v, _):
    f = 1
    return f * v

# Assemble local problem
dofs_global = Vh_global.N

Aps = []
Aps_inv = []
bs=[]
Ds=[]
initial_interface_conditions = []
b_global = np.zeros(dofs_global)



for i in range(0,nb_partition):
    A = a.assemble(Vhs[i])    
    b = l.assemble(Vhs[i])
    local_ext_boundary_nodes = ext_boundary_nodes[i]
    global_index =ovr_subdomain_to_global[i][local_ext_boundary_nodes]
    Ap = skfem.enforce(A, b,D=local_ext_boundary_nodes)
    Aps.append(Ap[0])
    Ap_inv = scipy.sparse.linalg.splu(Ap[0])
    Aps_inv.append(Ap_inv)
    b_global[ovr_subdomain_to_global[i]] += partition_of_unity[i]*Ap[1]


A = a.assemble(Vh_global)    
b = l.assemble(Vh_global)
A_global_mesh,b_global_mesh = skfem.enforce(A,b,D=np.array(boundary_nodes_global))
# print(f"A_global={A_global.shape}")

def A_global_matrix_vector_product(x: np.ndarray) -> np.ndarray:
    return global_matrix_vector_product(Aps,x,partition_of_unity,ovr_subdomain_to_global)

def A_apply_ASM_precondition(x: np.ndarray) -> np.ndarray:
    return apply_ASM_precondition(Aps_inv,x,ovr_subdomain_to_global,nb_partition,ext_boundary_nodes)

def A_apply_RAS_precondition(x: np.ndarray) -> np.ndarray:
    return apply_RAS_precondition(Aps_inv,x,ovr_subdomain_to_global,nb_partition,ext_boundary_nodes,partition_of_unity)


A = scipy.sparse.linalg.LinearOperator(shape=(dofs_global,dofs_global),matvec=A_global_matrix_vector_product,dtype="float64")

M_inv = scipy.sparse.linalg.LinearOperator(shape=(dofs_global,dofs_global),
matvec=A_apply_RAS_precondition,dtype="float64")

x,res = stationary_iterative_solver(A,b_global_mesh,M_inv,maxiter=10)
print(f"res stationary solver = {res}")
plot_res(res,"RAS.png",title="RAS")

x_gmres,info = scipy.sparse.linalg.gmres(A,b_global_mesh)
print(f"gmres info={info}")
res = np.linalg.norm(A@x - b_global_mesh)/np.linalg.norm(b_global_mesh)
print(f"res gmres = {res}")
# plot_res(res,"ASM.png",title="ASM")


dx = x - x_gmres
diff = np.linalg.norm(dx)/np.linalg.norm(x_gmres)
print(f"Realtive difference = {diff}")


# M_inv = scipy.sparse.linalg.LinearOperator(shape=(dofs_global,dofs_global),matvec=A_apply_ASM_precondition,dtype="float64")
