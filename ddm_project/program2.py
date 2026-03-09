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

    boundaries = gmsh.model.getEntities(dim=dim - 1)
    for i, (_, boundary_tag) in enumerate(boundaries):
        gmsh.model.addPhysicalGroup(dim - 1, [boundary_tag], name=f"boundary_{i}")
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

# Idendify mapping from interfaces
exchange_indices = [] # exchange_indices[i][p] gives the neighbour index and local index of same node for each boundary node
interface_nodes =[] # local indices of boundary nodes
boundary_nodes = [] # local indices for exterior boundary
print( submeshes[1].physical_group_elements)

for i in range(0,nb_partition):
    print(f"i={i}")
    interface_elem = submeshes[i].physical_group_elements[('interface', 0, None)]
    nodes_on_interface = np.unique(np.concatenate(interface_elem))
    exterior_boundary_elem = submeshes[i].physical_group_elements[(f"boundary_{i}", 0, 1+i)]
    nodes_on_exterior_boundary = np.unique(np.concatenate(exterior_boundary_elem))
    interface_nodes.append(nodes_on_interface)
    boundary_nodes.append(nodes_on_exterior_boundary)
    neigh = neighbors[i]
    inters = intersections[i]
    print(f"neigh = {neigh}")
    print(f"inters = {inters}")
    print(f"nodes_on_interface={nodes_on_interface}")
    args={}
    for j, n in enumerate(neigh):
        temp = np.zeros(len(inters[j]),dtype=int)
        other_neigh = neighbors[n]
        for jj,nn in enumerate(other_neigh):
            if nn == i:
                temp= intersections[n][jj]
        args[n] = temp
        exchange_indices.append(args)

print(f"interface_nodes={interface_nodes}")
print(f"exchange_indices={exchange_indices}")
# print(neighbors)
        
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
initial_interface_conditions = []

for i in range(0,nb_partition):
    A = a.assemble(Vhs[i])    
    b = l.assemble(Vhs[i])
    As.append(A)
    bs.append(b)
    nodes_on_interface = interface_nodes[i]
    initial_interface_conditions.append(np.zeros(A.shape[0]))
    initial_interface_conditions[i][nodes_on_interface] = 0.5
    xs.append([])

print(f"Initial interface conditions = {initial_interface_conditions}")

interface_conditions = initial_interface_conditions
xs_init = []
for iter in range(0,100):
    # Resolve the local problems    
    for i in range(0,nb_partition):
        local_boundary_nodes = np.hstack((interface_nodes[i],boundary_nodes[i]))
        Linear_system = skfem.enforce(As[i], bs[i],D=local_boundary_nodes,x=interface_conditions[i])
        x = skfem.solve(*Linear_system)
        xs[i] = x
        if iter == 0:
            xs_init.append(x) 
  
    for i in range(0,nb_partition):
        interface_conditions[i] = 0*interface_conditions[i]
        for j,n in enumerate(neighbors[i]):
            interface_conditions[i][intersections[i][j]] += xs[n][exchange_indices[i][n]]
    # Exchange information between sub-domains

# print(xs[0][exchange_indices[0]])

print(f"New interface conditions = {interface_conditions}")

fig1 = plt.figure()
ax = fig1.add_subplot()
ax.set_title(f"Finite element solution Initialisation")
ax.axis("equal")
for i in range(0,nb_partition):
    plot(Vhs[i], xs_init[i],ax=ax)
# ax3.show()
plt.show()


fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.set_title(f"Finite element solution Final")
ax2.axis("equal")
for i in range(0,nb_partition):
    plot(Vhs[i], xs[i],ax=ax2)
# ax3.show()
plt.show()
