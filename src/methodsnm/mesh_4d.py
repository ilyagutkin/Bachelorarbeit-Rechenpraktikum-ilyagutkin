from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh
from methodsnm.trafo import TesseraktTransformation

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

        self.points = np.array([array(mapping(i/M,j/N,k/K,l/L)) for k in range(K+1) for l in range(L+1) for j in range(N+1) for i in range(M+1) ])
        
        self.vertices = np.arange((M+1)*(N+1)*(K+1)*(L+1))
                              
        self.hypercells = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1] for i in range(M) for j in range(N) for k in range(K) for l in range(L)], dtype=int)
        
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
