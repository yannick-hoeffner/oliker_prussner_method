######################################################################################################
#
#   utility and plotting functions for the implementation of the oliker-prussner-method;
#   not all functions are actually used, some are only for debugging/testing;
#   for further details see the "main.py" file
#
#   by Yannick Höffner, Friedrich-Schiller-University Jena,
#   Bachelor Thesis WS 2024/25
#
#   based on the original paper from V.I.Oliker and L.D.Prussner:
#   "On the Numerical Solution of the Equation [...] and Its Discretizations, I"
#   Numerische Mathematik 54.3 (1989): 271-294
#
######################################################################################################
import sys
import os
import numpy as np

import matplotlib.pylab as plt
import matplotlib.tri as mtri


def bigprint(text):
    """makes a significant print in the command line

    Args:
        text (string): text to be printed with surrounding markers
    """
    print("\n#####################################################")
    print(text)
    print("#####################################################")


class HiddenPrints:
    """ disables prints inside the "with" section, for more details see
    https://stackoverflow.com/a/45669280/10546747
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def show(elements, coords, val, figureNumber=1, title=""):
    """plots the given function according to
    a triangulation and the coordinates of the
    triangulation vertices. Plots a triangle surface.

    Args:
        elements (ndarray): list of triangles given as indices in
            the coords list
        coords (ndarray): x,y coordinates
        val (ndarray): z-value which should be plot to the vertices
        title (string): title of the plot
    """
    tri = mtri.Triangulation(
        coords[:, 0], coords[:, 1], triangles=elements)
    fig = plt.figure(figureNumber)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    ax.plot_trisurf(coords[:, 0], coords[:, 1],
                    val, triangles=tri.triangles, cmap=plt.cm.Spectral)
    plt.plot([0], [0], [0])


def plot_neigbourhood(_index: int, _coords: np.ndarray, _values: np.ndarray, _adjNext: np.ndarray, fig_num=1, max_alpha=None, _alphas: np.ndarray = None, _angles: np.ndarray = None):
    """plots the given neighbourhood of a vertex with a potentially given alpha value

    Args:
        _index (int): the index of the vertex in the coords array
        _coords (np.ndarray): x,y coordinates given [[x1,y1], ...]
        _values (np.ndarray): values of the function at the vertices given as list with respect to the coords array
        _adjNext (np.ndarray): adjacency information of the neighbourhood, see 'optri.getNeighborhoodCW' or 'optri.getNeighborhoodCCW'
            for further information
        fig_num (int, optional): figure where to plot the neighbourhood. Defaults to 1.
        max_alpha (_type_, optional): if you want to visualize the point where two neigbouring
            triangles fall into one plane. Defaults to None.
        _alphas (np.ndarray, optional): alpha values for each neighbouring vertex
        _angles (np.ndarray, optional): alpha values for each neighbouring vertex
    """
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(f"Neighbourhood of vertex {_index}")

    ax.plot(_coords[_adjNext[1:-1], 0],
            _coords[_adjNext[1:-1], 1], _values[_adjNext[1:-1]], 'o')
    ax.plot(_coords[_index, 0], _coords[_index, 1], _values[_index], 'go')
    if max_alpha is not None and max_alpha != -np.inf:
        u_val_i = max_alpha
    else:
        u_val_i = _values[_index]
    ax.plot(_coords[_index, 0], _coords[_index, 1], u_val_i, 'ro')

    for i in range(1, _adjNext.size-1):
        # plot line to neighbouring vertex
        ax.plot([_coords[_index, 0], _coords[_adjNext[i], 0]], [
                _coords[_index, 1], _coords[_adjNext[i], 1]], [
                    u_val_i, _values[_adjNext[i]]], marker=" ", color="black")
        # plot line between neighbouring vertices
        ax.plot([_coords[_adjNext[i], 0], _coords[_adjNext[i+1], 0]], [
                _coords[_adjNext[i], 1], _coords[_adjNext[i+1], 1]], [
                    _values[_adjNext[i]], _values[_adjNext[i+1]]], marker=" ", color="black")
        text: str = ""
        if _alphas is not None:
            text = text + f"alp={_alphas[i-1]}"
        if _angles is not None:
            text = text + f" ang={_angles[i-1]}"
        if len(text) > 0:
            text = f"{i} " + text
            ax.text(_coords[_adjNext[i], 0],
                    _coords[_adjNext[i], 1], _values[_adjNext[i]], text)


def getNormalVectors(_triangles, _coordinates, _values):
    """calculates the normal vectors of all given triangles

    Args:
        _triangles (ndarray): list of vertices indices, describing the triangles
        _coordinates (ndarray): x,y coordinates of the vertices 
        _values (ndarray): z values of vertices

    Returns:
        N (ndarray): normal vectors for all triangles
    """
    hullP = np.vstack((_coordinates.T, [_values])).T
    xyz = hullP[_triangles]
    # second triangle point minus first triangle point
    p2Mp1 = xyz[:, 1]-xyz[:, 0]
    # third triangle point minus first triangle point
    p3Mp1 = xyz[:, 2]-xyz[:, 0]
    return np.cross(p2Mp1, p3Mp1, axis=1)


def pNormError(_solution, _u, p):
    """calculates the p norm difference between
    '_solution' and '_u'

    Args:
        _solution (ndarray): exact/reference value
        _u (ndarray): calculated value
        p (float): p norm selection

    Returns:
        pnorm (float): ||solution-u||_p
    """
    if p == np.inf:
        return np.max(np.abs(_solution-_u))
    return np.power(np.sum(np.power(np.abs(_solution-_u), p)), 1/p)


if __name__ == "__main__":
    print("this utillity routine is used to plot the final result of a latest_run.npz file")
    if len(sys.argv) < 2:
        print("usage: python oliker_prussner_plot.py <path>")
        sys.exit(1)

    path = sys.argv[1]

    data = np.load(path)
    calculated_triags = data["triags"]
    calculated_coords = data["coords"]
    calculated_u = data["u"]
    _sol = data["solution"]
    inf_norm = data["inf_norm"]

    plt.figure(100)
    show(calculated_triags, calculated_coords, calculated_u, figureNumber=100,
         title="calculated solution")
    plt.figure(101)
    show(calculated_triags, calculated_coords, np.abs(calculated_u-_sol), figureNumber=101,
         title="pointwise absolute distance to exact solution")
    plt.figure(102)
    plt.title("inf-norm-error per iterations")
    plt.plot(np.arange(1, inf_norm.size+1, 1),
             inf_norm, ls="None", marker="o")
    plt.yscale("log")

    plt.show()
