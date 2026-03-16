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

dim: int = 1
nb_partition: int = 4
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
submeshes, neighbors, intersections, partition_of_unity, _ = add_overlap(
        mesh_global, 1
    )

# Create local FEM meshes and basis
sk_submeshes = []
Vhs = []
for i in range(0,nb_partition):
    nodes = submeshes[i].nodes.T
    elements = submeshes[i].elements.T

    nodes =nodes[0,:]
    skmesh=skfem.MeshLine(nodes,elements)
    sk_submeshes.append(skmesh)
    Vhs.append(skfem.Basis(skmesh, skfem.ElementLineP1()))

# Idendify mapping from interfaces
exchange_indices = [] # exchange_indices[i][n] gives the indices of the overlap of submesh i with submesh n in the local numbering of submesh n. It corresponds with the overlap in  intersections[i] corresponding to neighbour n in neighbours[i]
overlap_nodes =[] # local indices of all overlap nodes
ext_boundary_nodes = [] # local indices for exterior boundary
boundary_nodes = [] # local indices for boundary nodes and overlap nodes

# Boundary tags for this problem
ext_boundary_tags = [("boundary_0", 0, 1),("boundary_1", 0, 2)]

# Create mapping
for i in range(0,nb_partition):
    # Obtain overlapping nodes between subdomains
    interface_elem = submeshes[i].physical_group_elements[('interface', 0, None)]
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
    nodes_on_interface = overlap_nodes[i]
    initial_interface_conditions.append(np.zeros(A.shape[0]))
    initial_interface_conditions[i][nodes_on_interface] = 0.5
    xs.append([])


# Specify initial conditions
interface_conditions = initial_interface_conditions

# Specify number of iterations
nb_iterations= 50



if nb_partition > 4:
    raise ValueError(f"Change markers to support more than 4 subdomains")

# Plotting parameters
markers= ["o","x","+","*"]
cmap = cm.viridis
norm = colors.Normalize(vmin=0, vmax=nb_iterations)
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.set_title(f"Finite element solution Final")
ax2.axis("equal")

print(exchange_indices[1])


for iter in range(0,nb_iterations):
    # Resolve the local problems    
    for i in range(0,nb_partition):
        local_ext_boundary_nodes = boundary_nodes[i]
        Linear_system = skfem.enforce(As[i], bs[i],D=local_ext_boundary_nodes,x=interface_conditions[i])
        x = skfem.solve(*Linear_system)
        xs[i] = x
        if iter == 0:
            xs_init.append(x) 

    # Exchange information between sub-domains
    for i in range(0,nb_partition):
        interface_conditions[i] = 0*interface_conditions[i]
        for j,n in enumerate(neighbors[i]):
            interface_conditions[i][intersections[i][j]] += partition_of_unity[n][exchange_indices[i][n]]*xs[n][exchange_indices[i][n]]

    # Plot solution
    for i in range(0,nb_partition):
        x = sk_submeshes[i].p[0]
        ind=np.argsort(x)
        ax2.plot(x[ind],xs[i][ind],marker=markers[i],color=cmap(norm(iter)))

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required for older matplotlib versions

fig2.colorbar(sm, ax=ax2, label="Iterations")


x_global =np.array(mesh_global.nodes)[:,0]
ind = np.argsort(x_global)
u_sol = 0.5*x_global*(1-x_global)

ax2.plot(x_global[ind],u_sol[ind],linestyle="dotted",color="k")

plt.show()


print(f'Testing global_matrix_vector_product')

from program5 import global_matrix_vector_product

y = global_matrix_vector_product(As,xs,partition_of_unity,neighbors,intersections,exchange_indices)
print(f"y = {y}")