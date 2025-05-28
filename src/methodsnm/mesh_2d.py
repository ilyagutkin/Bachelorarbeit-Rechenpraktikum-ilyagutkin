from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh
from methodsnm.trafo import TriangleTransformation

class Mesh2D(Mesh):
    def __init__(self):
        self.dimension = 2

class StructuredRectangleMesh(Mesh2D):
    def __init__(self, M, N, mapping = None):
        super().__init__()
        if mapping is None:
            mapping = lambda x,y: [x,y]
        self.points = np.array([array(mapping(i/M,j/N)) for j in range(N+1) for i in range(M+1)])
        self.vertices = np.arange((M+1)*(N+1))
        self.faces = np.array([[    j*(M+1)+i,     j*(M+1)+i+1, (j+1)*(M+1)+i] for i in range(M) for j in range(N)] + 
                              [[  j*(M+1)+i+1, (j+1)*(M+1)+i+1, (j+1)*(M+1)+i] for i in range(M) for j in range(N)], dtype=int)
        self.edges = np.array([[    j*(M+1)+i,     j*(M+1)+i+1] for j in range(N+1) for i in range(M) ] + 
                              [[    j*(M+1)+i,   (j+1)*(M+1)+i] for i in range(M+1) for j in range(N)] + 
                              [[(j+1)*(M+1)+i,     j*(M+1)+i+1] for i in range(M) for j in range(N)], dtype=int)
        self.bndry_vertices = [i for i in range(M)] + [M+j*(M+1) for j in range(N)] \
                              + [(N+1)*(M+1)-i-1 for i in range(M)] + [(N-j)*(M+1) for j in range(N)]
        self.bndry_edges = [i for i in range(M)] + [2*N*M+M+j for j in range(N)] \
                              + [N*M+M-1-i for i in range(M)] + [(N+1)*M+N-1-j for j in range(N)]
        offset = 2*M*N+M+N
        self.faces2edges = np.array([[offset+i+j*M, M*(N+1)+j+i*N, i+j*M] for i in range(M) 
                                                                          for j in range(N)] \
                           +[[i+(j+1)*M, offset+i+j*M, M*(N+1)+j+(i+1)*N] for i in range(M) 
                                                                          for j in range(N)], dtype=int)
        self.special_meshsize = np.sqrt(1/M**2 +1/N**2)

    def filter_bndry_points(self ,extreme_type, index):
        points = self.points[self.bndry_vertices]
    
        if extreme_type == "min":
            target_value = min(p[index] for p in points)
        elif extreme_type == "max":
            target_value = max(p[index] for p in points)
        else:
            raise ValueError("extreme_type must be either 'min' or 'max'")
        
        return [i for i in self.bndry_vertices if self.points[i][index] == target_value]


    def find_element(self, ip):
        def check_element_contains_point(points, ip , tol=1e-10):
            """
            Prüfe, ob Punkt `ip` im Dreieck `points` liegt.
            
            Parameters:
                ip: [x, y] – zu prüfender Punkt
                points: 3x2-Array – drei Dreieckspunkte [[x0, y0], [x1, y1], [x2, y2]]
                tol: Toleranz für numerische Robustheit
            
            Returns:
                True oder False
            """
            a = np.array(points[0])
            b = np.array(points[1])
            c = np.array(points[2])
            p = np.array(ip)

            v0 = c - a
            v1 = b - a
            v2 = p - a

            M = np.column_stack((v1, v0))
            try:
                l1_l2 = np.linalg.solve(M, v2)
            except np.linalg.LinAlgError:
                return False  # Degeneriertes Dreieck

            λ1, λ2 = l1_l2
            λ0 = 1 - λ1 - λ2

            return (λ0 >= -tol) and (λ1 >= -tol) and (λ2 >= -tol)
            # Find the element that contains the point ip
            # This is a simple linear search, which can be improved with a more efficient algorithm
            
        for el in range(len(self.faces)):
            points = self.points[self.faces[el]]
            if check_element_contains_point(points,ip):
                return el
        raise Exception("point outside mesh")
    
    def trafo(self, elnr, codim=0, bndry=False):
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented yet")
        return TriangleTransformation(self, elnr)
