import numpy as np
import pyvista as pv


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def sample_points_3d(mesh, t_value, n=30):
    mins = mesh.points.min(axis=0)
    maxs = mesh.points.max(axis=0)

    xs = np.linspace(mins[0], maxs[0], n)
    ys = np.linspace(mins[1], maxs[1], n)
    zs = np.linspace(mins[2], maxs[2], n)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), np.full(X.size, t_value)])
    return (X, Y, Z), pts


def eval_abs(uvec, pts4):
    out = np.zeros(len(pts4))
    for i,p in enumerate(pts4):
        v = uvec._evaluate(p)
        out[i] = np.linalg.norm(v)
    return out


def eval_comp(uvec, pts4, comp):
    out = np.zeros(len(pts4))
    for i,p in enumerate(pts4):
        v = uvec._evaluate(p)
        out[i] = v[comp]
    return out


# ------------------------------------------------------------
# Animation Helper
# ------------------------------------------------------------

def make_grid(X, Y, Z, vals):
    grid = pv.StructuredGrid()
    grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nx, ny, nz = X.shape
    grid.dimensions = (nx, ny, nz)
    grid["u"] = vals
    return grid


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def animate_u_abs(uvec, t_list, samples=25, gif_name="u_abs.gif"):
    """ Animation des Betrags |u| über die Zeit t_list """
    mesh = uvec.mesh
    plotter = pv.Plotter(off_screen=True)

    X, Y, Z, grid = None, None, None, None
    first = True

    frames = []

    for t in t_list:
        (X, Y, Z), pts4 = sample_points_3d(mesh, t, samples)
        vals = eval_abs(uvec, pts4)

        grid = make_grid(X, Y, Z, vals)

        if first:
            plotter.add_volume(grid, opacity="sigmoid", name="vol")
            first = False
        else:
            plotter.update_volume(grid, name="vol")

        img = plotter.screenshot(return_img=True)
        frames.append(img)

    plotter.close()

    pv.save_gif(gif_name, frames)


def animate_u_component(uvec, comp, t_list, samples=25, gif_name="u_comp.gif"):
    """ Animation einer einzelnen Komponente u_comp über die Zeit """
    mesh = uvec.mesh
    plotter = pv.Plotter(off_screen=True)

    first = True
    frames = []

    for t in t_list:
        (X, Y, Z), pts4 = sample_points_3d(mesh, t, samples)
        vals = eval_comp(uvec, pts4, comp)

        grid = make_grid(X, Y, Z, vals)

        if first:
            plotter.add_volume(grid, opacity="sigmoid", name="vol")
            first = False
        else:
            plotter.update_volume(grid, name="vol")

        img = plotter.screenshot(return_img=True)
        frames.append(img)

    plotter.close()
    pv.save_gif(gif_name, frames)

def interactive_u_abs(uvec, t_min, t_max, samples=25):
    """Interaktive 3D-Visualisierung von |u| im EXTERNEN PyVista-Fenster."""
    mesh = uvec.mesh
    pl = pv.Plotter()   # <-- EXTERNES FENSTER

    # Startzustand
    (X, Y, Z), pts4 = sample_points_3d(mesh, t_min, samples)
    vals = eval_abs(uvec, pts4)
    grid = make_grid(X, Y, Z, vals)

    actor = pl.add_volume(grid, opacity="sigmoid", name="vol")

    def update_t(t):
        nonlocal actor
        (X, Y, Z), pts4 = sample_points_3d(mesh, t, samples)
        vals = eval_abs(uvec, pts4)
        new_grid = make_grid(X, Y, Z, vals)

        pl.remove_actor(actor)
        actor = pl.add_volume(new_grid, opacity="sigmoid", name="vol")

    pl.add_slider_widget(
        update_t,
        rng=[t_min, t_max],
        value=t_min,
        title="t",
    )

    pl.show()   # <-- öffnet ein eigenes Fenster (nicht Notebook)
    


def interactive_u_component(uvec, comp, t_min, t_max, samples=25):
    """Interaktive 3D-Visualisierung einer Komponente u[comp] im EXTERNEN Fenster."""
    mesh = uvec.mesh
    pl = pv.Plotter()

    (X, Y, Z), pts4 = sample_points_3d(mesh, t_min, samples)
    vals = eval_comp(uvec, pts4, comp)
    grid = make_grid(X, Y, Z, vals)

    actor = pl.add_volume(grid, opacity="sigmoid", name="vol")

    def update_t(t):
        nonlocal actor
        (X, Y, Z), pts4 = sample_points_3d(mesh, t, samples)
        vals = eval_comp(uvec, pts4, comp)
        new_grid = make_grid(X, Y, Z, vals)

        pl.remove_actor(actor)
        actor = pl.add_volume(new_grid, opacity="sigmoid", name="vol")

    pl.add_slider_widget(
        update_t,
        rng=[t_min, t_max],
        value=t_min,
        title=f"t (component {comp})",
    )

    pl.show()