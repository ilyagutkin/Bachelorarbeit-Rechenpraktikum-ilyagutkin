from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from netgen.csg import CSGeometry, OrthoBrick
from ngsolve import VOL, BND, Mesh as NGSMesh
from methodsnm.mesh import Mesh
from methodsnm.trafo import TesseraktTransformation , HypertriangleTransformation

class Mesh4D(Mesh):
    def __init__(self):
        self.dimension = 4

class StructuredTesseraktMesh(Mesh4D):
    def __init__(self, M, N , K, L, mapping = None):
        super().__init__()

        def check_bndryedges(bndry_vertices, edge):
            if set(edge).issubset(set(bndry_vertices)):
                return True
            else:
                return False
            
        def check_bndryfaces(bndry_vertices, face):
            if set(face).issubset(set(bndry_vertices)):
                return True
            else:
                return False
        
        def check_bndryvolumes(bndry_vertices, volume):
            if set(volume).issubset(set(bndry_vertices)):
                return True
            else:
                return False

        def rekursiv_bndry(lis):
            if len(lis) == 1:
                return [0,lis[0]]
            else:
                produkt1 = np.math.prod([x + 1 for x in lis])
                first = lis.pop(0)
                produkt = np.math.prod([x + 1 for x in lis])
                bndry_vertices = rekursiv_bndry(lis)
                #print(liste)
                liste = []
                liste += bndry_vertices
                liste += [i + j*produkt for i in bndry_vertices for j in range(first)]
                liste += [produkt1-1-i for i in range(produkt)]
                liste += [i for i in range(produkt)]
                return list(set(liste))
            
        
        if mapping is None:
            mapping = lambda x,y,z,t: [x,y,z,t]

        self.points = np.array([array(mapping(i/M,j/N,k/K,l/L)) for l in range(L+1) for k in range(K+1) for j in range(N+1) for i in range(M+1) ])
        
        self.vertices = np.arange((M+1)*(N+1)*(K+1)*(L+1))
                              
        self.hypercells = np.array([[    l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i+1,
                                         l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                         l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i+1,
                                         l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1] for i in range(M) for j in range(N) for k in range(K) for l in range(L)], dtype=int)
        
        self.volumes = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1]for i in range(M) for j in range(N) for k in range(K)   for l in range(L+1)]
                                     +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1]for i in range(M) for j in range(N) for k in range(K+1)   for l in range(L)]
                                     +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                        l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1,
                                        (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                        (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K)   for l in range(L)]
                                        +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i]for i in range(M+1) for j in range(N) for k in range(K)   for l in range(L)], dtype=int)
        
        self.faces = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1]for i in range(M) for j in range(N) for k in range(K+1)   for l in range(L+1)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                        l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K)   for l in range(L+1)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i]for i in range(M+1) for j in range(N) for k in range(K)   for l in range(L+1)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1, 
                                   (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K+1)   for l in range(L)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                   (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i] for i in range(M+1) for j in range(N) for k in range(K+1)   for l in range(L)]    
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,
                                   (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i] for i in range(M+1) for j in range(N+1) for k in range(K)   for l in range(L)], dtype=int)
        
        self.edges = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K+1)   for l in range(L+1)]
                              +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i]   for i in range(M+1) for j in range(N) for k in range(K+1)   for l in range(L+1)]
                              +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i]   for i in range(M+1) for j in range(N+1) for k in range(K)   for l in range(L+1)]
                              +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i] for i in range(M+1) for j in range(N+1) for k in range(K+1)   for l in range(L)], dtype=int)

        self.bndry_vertices = np.array(rekursiv_bndry([L,K,N,M]))
        self.bndry_edges = [i for i in self.edges if check_bndryedges(self.bndry_vertices,i) ]
        self.bndry_faces = [i for i in self.faces if check_bndryfaces(self.bndry_vertices,i) ]
        self.bndry_volumes = [i for i in self.volumes if check_bndryvolumes(self.bndry_vertices,i) ]
        self.xlength = M
        self.ylength = N    
        self.zlength = K
        self.tlength = L
        self.special_meshsize = 1/min(M,N,K,L)

    def filter_bndry_points(self ,extreme_type, index):
        points = self.points[self.bndry_vertices]
    
        if extreme_type == "min":
            target_value = min(p[index] for p in points)
        elif extreme_type == "max":
            target_value = max(p[index] for p in points)
        else:
            raise ValueError("extreme_type must be either 'min' or 'max'")
        
        return [i for i in self.bndry_vertices if self.points[i][index] == target_value]

    def trafo(self, elnr, codim=0, bndry=False):
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented yet")
        return TesseraktTransformation(self, elnr)


class UnstructuredHypertriangleMesh(Mesh4D):
    def generate_3d_mesh(maxh=0.2):
        # Erstelle eine einfache 3D-Geometrie Ω (hier: Einheitswürfel)
        geo = CSGeometry()
        cube = OrthoBrick(*[(0, 0, 0), (1, 1, 1)]).bc("outer")
        geo.Add(cube)

        # Generiere 3D-Mesh
        ngmesh = geo.GenerateMesh(maxh=maxh)
        type(ngmesh)
        ngmesh.Curve(1)  # order = 1 (linear)
        
        return NGSMeshMesh(ngmesh)

    def __init__(self, T, ngmesh=None ):
        if ngmesh is None:
            ngmesh = self.generate_3d_mesh()
        super().__init__()
        nv = ngmesh.nv
        self.vertices = np.arange((T+1)*nv)
        self.hypercells = np.array([[t*nv+el.vertices[0].nr, t*nv+el.vertices[1].nr, t*nv+el.vertices[2].nr, t*nv+el.vertices[3].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   +[[t*nv+el.vertices[0].nr, t*nv+el.vertices[1].nr, t*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   +[[t*nv+el.vertices[0].nr, t*nv+el.vertices[1].nr, (t+1)*nv+el.vertices[1].nr, (t+1)*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   +[[t*nv+el.vertices[0].nr, (t+1)*nv+el.vertices[0].nr, (t+1)*nv+el.vertices[1].nr, (t+1)*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   , dtype=int)
        
        vertex_indices = [None] *((T+1)*nv +1)

        for el in ngmesh.Elements(VOL):
            for v in el.vertices:
                    p = ngmesh[v].point
                    nr = v.nr
                    if vertex_indices[nr] is None:
                        vertex_indices[nr] = (p[0], p[1], p[2])

        self.points = np.array([np.append(p, t/T)for t in range(T+1)for p in vertex_indices if p is not None])

        boundary_vertices = list(dict.fromkeys(v.nr for el in ngmesh.Elements(BND) for v in el.vertices))
        boundary_vertices = list(dict.fromkeys(boundary_vertices))
        main_list = [t*nv+v for t in range(T) for v in boundary_vertices]
        first = list(range(nv))
        last = self.vertices[-nv:].tolist()
        self.bndry_vertices = np.array(list(set(main_list + first + last)))
        self.special_meshsize = 1/T
        self.initial_bndry_vertices = first
        self.top_bndry_vertices = last

    def trafo(self, elnr, codim=0, bndry=False):
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented yet")
        return HypertriangleTransformation(self, elnr)