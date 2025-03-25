from numpy import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.webgui import Draw
from netgen.occ import *

def CG_solver(mesh,order,w,eps):
    fes = H1(mesh, order=order, dirichlet="bottom|right|left")
    u,v = fes.TnT()
    n = specialcf.normal(2)
    Au = CoefficientFunction((grad(u)[0],0))
    B = CoefficientFunction((w,1))
    a = BilinearForm(fes, symmetric=False)
    a += Au*grad(v)*dx 
    a += (B*grad(u))*v*dx
    a.Assemble()

    f = LinearForm(fes)
    f += 0*v*dx
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.Set(exp(w*x/2)*sin(pi*x),BND)  # initial condition

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * res
    return(gfu)

def DG_solver(mesh,order,w,eps):
    fes = L2(mesh, order=order, dgjumps=True)
    u,v = fes.TnT()
    lam = 2
    h = specialcf.mesh_size

    dS = dx(skeleton=True) 
    jump_u = u-u.Other()
    jump_v = v-v.Other()
    n = specialcf.normal(2)
    avgu = 0.5*n[0] * (grad(u)[0]+grad(u.Other())[0])
    avgv= 0.5*n[0] * (grad(v)[0]+grad(v.Other())[0])

    Au = CoefficientFunction((grad(u)[0],0))
    B = CoefficientFunction((w,1))

    diff = Au*grad(v)*dx # diffusion term
    diff += -avgu * jump_v *dS #term from the partial integration
    #diff += (B*grad(u))*v*dx # time derivative term + convectionterm
    diff += -(n[0]*grad(u)[0]*v)* ds(skeleton=True) #term from the partial integration
    diff += lam*order**2/h*u*v * ds(skeleton=True) # penalty term
    diff += lam*order**2/h*jump_u*jump_v*dS # penalty term
    uhat = IfPos(B*n, u, u.Other())
    con = -B*grad(v)*u*dx
    con += (B*n)*uhat *jump_v*dS
    diffcon = BilinearForm(diff + con).Assemble()
    
    f = LinearForm(fes)
    f += lam*order**2/h*exp(w*x/2)*sin(pi*x)*v*ds(skeleton=True, definedon=mesh.Boundaries("bottom"))#weakly impose boundary condition
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data += diffcon.mat.Inverse(fes.FreeDofs()) * f.vec
    return gfu

def SUPG_solver(mesh,order,w,eps):
    fes = H1(mesh, order=order, dirichlet="bottom|right|left")
    u,v = fes.TnT()
    n = specialcf.normal(2)
    h = specialcf.mesh_size

    Cinv = order**4 # depends on order in general
    diffusion = eps

    gamma_0 = 1
    #gamma_T = gamma_0 * h**2 / (w * order**2)
    gamma_T = gamma_0 * h / sqrt(w**2 + Cinv * diffusion)
    #gamma_T = 0

    ws = CF((w,1))

    def gradx(u):
        return grad(u)[0]
    def dt(u):
        return grad(u)[1]

    uxx = u.Operator("hesse")[0,0]

    a = BilinearForm(fes, symmetric=False)
    a += gradx(u)*gradx(v)*dx + (ws*grad(u))*v*dx
    a += gamma_T * (-uxx + ws * grad(u))*(ws* grad(v)) * dx
    a.Assemble()

    f = LinearForm(fes)
    f += 0*v*dx
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.Set(exp(w*x/2)*sin(pi*x),BND)  # initial condition
    Draw(gfu)

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * res
    return(gfu)

def maxy(mesh,deformation):
        max_y_difference = 0  # Variable zur Speicherung der größten y-Differenz
        max_x_difference = 0  # Variable zur Speicherung der größten x-Differenz
        # Iteriere über alle Dreiecke im Mesh
        for el in mesh.Elements():
                vertices = [tuple(deformation(mesh[vi].point[0], mesh[vi].point[1])) for vi in el.vertices]  # Eckpunkte des Dreiecks
                y_coords = [p[1] for p in vertices]  # Nur die y-Koordinaten extrahieren
                x_coords = [p[0] for p in vertices]  # Nur die x-Koordinaten extrahieren

                y_difference = max(y_coords) - min(y_coords)  # Höchster - niedrigster Wert
                max_y_difference = max(max_y_difference, y_difference)  # Maximale Differenz speichern

                x_difference = max(x_coords) - min(x_coords)  # Höchster - niedrigster Wert
                max_x_difference = max(max_x_difference, x_difference)  # Maximale Differenz speichern
        return max_y_difference, max_x_difference

Meshes =[]
for p in range(-3,1):
        for i in range(-3,1):
                L = 2**i
                T = 2**(p) # length of the time interval
                shape = Rectangle(L,T).Face()
                shape.edges.Min(X).name="left"
                shape.edges.Max(X).name="right"
                shape.edges.Min(Y).name="bottom"
                shape.edges.Max(Y).name="top"
                mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.05))
                deformation = GridFunction(H1(mesh,order=1,dim=mesh.dim))
                deformation.Set(CF((2**-i*x,2**(-p)*y))-CF((x,y)))
                mesh.SetDeformation(deformation)
                maximal_y_difference, maximal_x_difference = maxy(mesh,deformation)
                Meshes.append((mesh,maximal_y_difference,maximal_x_difference))

Solver =[CG_solver, DG_solver, SUPG_solver]
w = 1
eps =1

for order in range(3):
    for solver in Solver:
        for mesh in Meshes:
            gfu = solver(mesh[0],order,w,eps)
            L2fehler = sqrt(Integrate((gfu-(sin(pi*x)*exp(-(pi**2+w**2/4)*y)*exp(w*x/2)))**2, mesh[0]))
            print(f"Methode: {solver.__name__} , order ={order} L2 Fehler: {L2fehler} max_y ={mesh[1]} max_x={mesh[2]}" )
        