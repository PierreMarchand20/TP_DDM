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
boundary_nodes =[] # local indices of boundary nodes
for i in range(0,nb_partition):
    print(f"i={i}")
    boundary_elem = submeshes[i].physical_group_elements[('interface', 0, None)]
    nodes_on_interface = np.unique(np.concatenate(boundary_elem))
    boundary_nodes.append(nodes_on_interface)
    neigh = neighbors[i]
    inters = intersections[i]
    print(f"neigh = {neigh}")
    print(f"inters = {inters}")
    print(f"nodes_on_interface={nodes_on_interface}")
    args={}
    for node in nodes_on_interface:
        for j,n in enumerate(neigh):
            # print(f"j,n = {j},{n}")
            # print(inters[j])
            for jj in range(0,len(inters[j])):
                if node == inters[j][jj]:
                    other_inter = intersections[n]
                    other_neigh = neighbors[n]
                    for jjj,nn in enumerate(other_neigh):
                        if nn == i:
                            other_node = other_inter[jjj][jj]
                            args[node]=(n,other_node)
                    break
            # print(intersections[n])
        exchange_indices.append(args)

print(f"boundary_nodes={boundary_nodes}")
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
    boundary_elem = submeshes[i].physical_group_elements[('interface', 0, None)]
    nodes_on_interface = np.unique(np.concatenate(boundary_elem))
    initial_interface_conditions.append(np.zeros(A.shape[0]))
    initial_interface_conditions[i][nodes_on_interface] = 0.5
    xs.append([])

print(f"Initial interface conditions = {initial_interface_conditions}")

interface_conditions = initial_interface_conditions
for iter in range(0,5):
    # Resolve the local problems    
    for i in range(0,nb_partition):
        Linear_system = skfem.enforce(A, b,D=nodes_on_interface,x=interface_conditions[i])
        x = skfem.solve(*Linear_system)
        xs[i] = x
    for i in range(0,nb_partition):
        for p in boundary_nodes[i]:
            n,j = exchange_indices[i][p]
            interface_conditions[i][p] = xs[n][j]
    # Exchange information between sub-domains

# print(xs[0][exchange_indices[0]])

print(f"New interface conditions = {interface_conditions}")