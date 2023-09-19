from methodsnm.fe_1d import *
from methodsnm.fe_2d import *
from methodsnm.meshfct import *
import numpy as np
import matplotlib.pyplot as plt       
#import pylab as plt
from math import ceil, sqrt

def DrawSegmentFE(fe, sampling=100, xkcd_style=False, derivative=False):
    if not isinstance(fe, FE_1D):
        raise ValueError("fe must be an instance of FE_1D")
    xvals = np.linspace(0, 1, sampling).reshape((sampling, 1))
    if derivative:
        yvals = np.array([fe.evaluate_deriv(xi) for xi in xvals])
    else:
        yvals = fe.evaluate(xvals) 
        #yvals = np.array([fe.evaluate(xi) for xi in xvals])
    plt.style.use("fivethirtyeight")
    if xkcd_style:
        plt.xkcd()
    plt.plot(xvals, yvals)
    plt.legend(["$\phi_{"+str(i)+"}$" for i in range(fe.ndof)])
    plt.show()

# identify best subplot pattern:
def identify_best_subplot_pattern(L):
    """
    Given the number of subplots L, this function identifies the best subplot pattern
    to use for a plot with L subplots. The function returns the number of rows and columns
    for the subplot grid.
    
    Args:
    - L: int, the number of subplots
    
    Returns:
    - N: int, the number of rows in the subplot grid
    - M: int, the number of columns in the subplot grid
    """
    patterns = [(1,1),(2,1),(3,1),(3,2),(3,3),(5,2)]
    s_of_p = [None for p in patterns]
    D_of_p = [None for p in patterns]
    for pi, p in enumerate(patterns):
        ps = p[0] * p[1]
        D = int(ceil(sqrt(L/ps)))
        s_of_p[pi] = ps * D**2
        D_of_p[pi] = D
    mini = s_of_p.index(min(s_of_p))
    pM,pN = patterns[mini]
    D = D_of_p[mini]
    N,M = pN*D, pM*D
    return N,M


def DrawTriangleFE(fe, sampling=10):
    x0vals = np.array([0,1,0,0])
    y0vals = np.array([0,0,1,0])

    xvals = np.array([j/sampling for i in range(sampling+1) for j in range(sampling+1-i)])
    yvals = np.array([i/sampling for i in range(sampling+1) for j in range(sampling+1-i)])

    # fe vals on grid
    fevals = np.array([fe.evaluate(np.array([xx,yy])) for (xx,yy) in zip(xvals,yvals)])

    n_grid_pts = (sampling+1)*(sampling+2)//2

    # map from (i,j) to n and vice versa:
    cnt = 0
    n2ij = []
    ij2n = [[None for j in range(sampling+1)] for i in range(sampling+1)]
    for i in range(sampling+1):
        for j in range(sampling+1-i):
            n2ij.append((i,j))
            ij2n[i][j] = cnt
            cnt += 1

    # triangles for plotting:
    trigs = [[ij2n[i][j], ij2n[i][j+1], ij2n[i+1][j]] for j in range(sampling) for i in range(sampling-j)]
    trigs += [[ij2n[i][j+1], ij2n[i+1][j+1], ij2n[i+1][j]] for j in range(sampling-1) for i in range(sampling-j-1)]

    N,M = identify_best_subplot_pattern(fe.ndof)
    for dof in range(fe.ndof):
        m = dof % M; n = dof // M
        ax = plt.subplot2grid((N,M),(n, m), projection='3d')
        for i in range(3):
            ax.plot(x0vals[i:i+2], y0vals[i:i+2], linewidth=2.0, color="black", antialiased=True)
        ax.plot_trisurf(xvals, yvals, trigs, fevals[:,dof], cmap=plt.cm.Spectral, linewidth=0.0, antialiased=True)
    plt.show()

def DrawMesh1D(mesh):
    x_v = mesh.vertices
    plt.plot(x_v,np.zeros(len(x_v)),'|',label='vertices')
    plt.xlabel("x")
    plt.legend()
    plt.show()

def DrawFunction1D(f, sampling = 10, mesh = None, show_mesh = False):
    if not isinstance(f, list):
        DrawFunction1D([f], sampling, mesh=mesh, show_mesh=show_mesh)
        return
    if not all([isinstance(fi, MeshFunction) for fi in f]):
        raise ValueError("f must be a list of MeshFunction instances")
    if mesh is None:
        mesh = f[0].mesh
    xy = []
    for elnr,vs in enumerate(mesh.elements()):
        trafo = mesh.trafo(elnr)
        xl, xr = mesh.vertices[vs]
        xy += [[xl*(1-ip_x) + xr*ip_x] + [fi.evaluate(np.array([ip_x]),trafo) for fi in f] for ip_x in np.linspace(0,1,sampling)]
    xy = np.array(xy)
    plt.plot(xy[:,0],xy[:,1::],'-')
    plt.xlabel("x")
    #plt.legend()
    if show_mesh:
        plt.plot(mesh.vertices,np.zeros(len(mesh.vertices)),'|',label='vertices')    
    plt.show()

def DrawShapes(fes):
   uhs = [FEFunction(fes) for i in range(fes.ndof)]
   for i in range(fes.ndof):
       uhs[i].vector[i] = 1
   DrawFunction1D(uhs)   

if __name__ == "__main__":
    p1 = P1_Segment_FE()
    DrawSegmentFE(p1, sampling=10)
    p1 = P1_Triangle_FE()
    DrawTriangleFE(p1, sampling=10)

