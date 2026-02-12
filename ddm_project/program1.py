import gmsh
import skfem
import numpy as np 
from skfem.helpers import dot, grad  
from skfem.visuals.matplotlib import plot
from matplotlib import pyplot as plt


from ddm_playground.mesh.gmsh import GmshContextManager, GmshOptions
from ddm_playground.mesh.plot import plot_mesh, plot_submesh

dim: int = 2
nb_partition: int = 2
gmsh_options = GmshOptions(mesh_name="mesh")

with GmshContextManager(gmsh_options) as mesh_generator:
    
    lc = 0.1  # characteristic length (mesh size)

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
    mesh = mesh_generator.generate(dim, nb_partition)

# matplotlib visualization
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_title(f"{dim}D Mesh from GMSH")
ax1.axis("equal")

plot_mesh(ax1, mesh)


nodes = mesh.nodes.T
elements = mesh.elements.T

nodes =nodes[0:2,:]
skmesh = skfem.MeshTri(nodes,elements)

ax2 = fig.add_subplot(122)
ax2.set_title(f"{dim}D Mesh from scikit-fem")
ax2.axis("equal")

ax2.plot(skmesh.p[0], skmesh.p[1], 'ok')
for t in skmesh.t.T: # transpose to iterate over columns
    ax2.plot(skmesh.p[0,[t[0],t[1]]], skmesh.p[1,[t[0],t[1]]], 'k')  # from vertex 0 to 1
    ax2.plot(skmesh.p[0,[t[1],t[2]]], skmesh.p[1,[t[1],t[2]]], 'k')  # from vertex 1 to 2
    ax2.plot(skmesh.p[0,[t[2],t[0]]], skmesh.p[1,[t[2],t[0]]], 'k')  # from vertex 2 back to 0


plt.show()
basis = skfem.Basis(skmesh, skfem.ElementTriP1())

# Define bilinear and linear form
@skfem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))

@skfem.LinearForm
def l(v, w):
    x, y = w.x  # global coordinates
    f = np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v

# Assemble matrices and essential boundary conditions
A = a.assemble(basis)
b = l.assemble(basis)
D = basis.get_dofs()

# Resolve system
x = skfem.solve(*skfem.condense(A, b, D=D))

@skfem.Functional
def error(w):
    x, y = w.x
    uh = w['uh']
    u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2. * np.pi ** 2)
    return (uh - u) ** 2

err=error.assemble(basis, uh=basis.interpolate(x))
print(f"Absolute error = {err}")
plt.show()

fig2 = plt.figure()
ax3 = fig2.add_subplot()
ax3.set_title(f"Finite element solution")
ax3.axis("equal")
plot(basis, x,ax=ax3)
# ax3.show()
plt.show()