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
import numpy as np
import numpy.typing as npt

import matplotlib.pylab as plt
import matplotlib.tri as mtri


def show(triangles: npt.NDArray[np.int64], coordinates: npt.NDArray[np.float64],
         z_values: npt.NDArray[np.float64], figureNumber: np.int64 = 1, title: str = "") -> None:
    """plots the given function according to
    a triangulation and the coordinates of the
    triangulation vertices. Plots a triangle surface.

    Args:
        triangles (NDArray[int64]): list of triangles given as indices in
            the coords list
        coordinates (NDArray[float64]): x,y coordinates
        z_values (NDArray[float64]): z-value which should be plot to the vertices
        figureNumber (int64, optional): figure number where to plot the function.
            Defaults to 1.
        title (string, optional): title of the plot. Defaults empty string.
    """
    tri = mtri.Triangulation(
        coordinates[:, 0], coordinates[:, 1], triangles=triangles)
    fig = plt.figure(figureNumber)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    ax.plot_trisurf(coordinates[:, 0], coordinates[:, 1],
                    z_values, triangles=tri.triangles, cmap=plt.cm.Spectral)
    plt.plot([0], [0], [0])


def plot_neigbourhood(index: np.int64, coordinates: npt.NDArray[np.float64],
                      z_values: npt.NDArray[np.float64], adjNext: npt.NDArray[np.int64],
                      fig_num: np.int64 = 1, max_alpha: np.float64 = None, alphas: npt.NDArray[np.float64] = None,
                      angles: npt.NDArray[np.float64] = None) -> None:
    """plots the given neighbourhood of a vertex with a potentially given alpha value

    Args:
        index (int64): the index of the vertex in the coords array
        coordinates (NDArray[float64]): x,y coordinates given [[x1,y1], ...]
        z_values (NDArray[float64]): values of the function at the vertices given as list with respect to the coords array
        adjNext (NDArray[int64]): adjacency information of the neighbourhood, see 'optri.getNeighborhoodCW' or 'optri.getNeighborhoodCCW'
            for further information
        fig_num (int64, optional): figure where to plot the neighbourhood. Defaults to 1.
        max_alpha (float64, optional): if you want to visualize the point where two neigbouring
            triangles fall into one plane. Defaults to None.
        alphas (NDArray[float64], optional): alpha values for each neighbouring vertex
        angles (NDArray[float64], optional): alpha values for each neighbouring vertex
    """
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(f"Neighbourhood of vertex {index}")

    ax.plot(coordinates[adjNext[1:-1], 0],
            coordinates[adjNext[1:-1], 1], z_values[adjNext[1:-1]], 'o')
    ax.plot(coordinates[index, 0],
            coordinates[index, 1], z_values[index], 'go')
    if max_alpha is not None and max_alpha != -np.inf:
        u_val_i = max_alpha
    else:
        u_val_i = z_values[index]
    ax.plot(coordinates[index, 0], coordinates[index, 1], u_val_i, 'ro')

    for i in range(1, adjNext.size-1):
        # plot line to neighbouring vertex
        ax.plot([coordinates[index, 0], coordinates[adjNext[i], 0]], [
                coordinates[index, 1], coordinates[adjNext[i], 1]], [
                    u_val_i, z_values[adjNext[i]]], marker=" ", color="black")
        # plot line between neighbouring vertices
        ax.plot([coordinates[adjNext[i], 0], coordinates[adjNext[i+1], 0]], [
                coordinates[adjNext[i], 1], coordinates[adjNext[i+1], 1]], [
                    z_values[adjNext[i]], z_values[adjNext[i+1]]], marker=" ", color="black")
        text: str = ""
        if alphas is not None:
            text = text + f"alp={alphas[i-1]}"
        if angles is not None:
            text = text + f" ang={angles[i-1]}"
        if len(text) > 0:
            text = f"{i} " + text
            ax.text(coordinates[adjNext[i], 0],
                    coordinates[adjNext[i], 1], z_values[adjNext[i]], text)
