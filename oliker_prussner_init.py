#####################################################################################################
#
#   initialisation of the oliker-prussner-method
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
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import oliker_prussner_plot as opplot
import oliker_prussner_triangulation as optri
import os
import sys


def f(val):
    """function for the right hand side of
    the Monge-Ampère-Equation

    Args:
        val (ndarray): array [[x1,y1],[x2,y2],...] with data from the domain

    Returns:
        ndarray: f(x,y) for every (x,y) pair as a list
    """
    if FUNCTION == 1:
        # parabola
        return np.repeat(4, np.size(val, 0))
    elif FUNCTION == 2:
        # tangens of the parbola
        return (16*(np.square(val[:, 0]) + np.square(val[:, 1]))*(np.tan(np.square(val[:, 0]) + np.square(
            val[:, 1])))+16)*np.power(np.cos(np.square(val[:, 0]) + np.square(val[:, 1])), -4)
    elif FUNCTION == 3:
        # abs function does not have an interpretation for f
        return np.repeat(0, np.size(val, 0))
    elif FUNCTION == 4:
        # exponential parabola
        return np.exp((np.square(val[:, 0]) + np.square(val[:, 1]))) * (val[:, 0]*val[:, 0]+val[:, 1]*val[:, 1]+1)
    elif FUNCTION == 5:
        # sin fraction
        return (np.sin(val[:, 0])**2*np.sin(val[:, 1])**2*(2-np.sin(val[:, 0])*np.sin(val[:, 1])))/(np.sin(val[:, 0])+np.sin(val[:, 1]))**4
    else:
        raise ValueError("function not implemented")


def computeMuFromGrid(_adjacancies, _triangles, _countADJ, _coordinates, _boundaryMask):
    """calculates a possible measure mu from an initial grid by setting the
    local measure of a vertex to the integral over the classical FEM hat functions
    on adjoint triangles

    Args:
        _adjacancies (ndarray): adjacencies list of initial triangulation;
            see optri.findAdjacencies for further information
        _triangles (ndarray): triangles list of initial triangulation;
            see optri.findAdjacencies for further information
        _countADJ (ndarray): adjacencies count of initial triangulation;
            see optri.findAdjacencies for further information
        _coordinates (ndarray): array [[x1,y1],[x2,y2],...] with coordinate data of
            the vertices
        _boundaryMask (ndarray): mask for _coordinates with 1 if the desired vertex
            is an boundary point and 0 if it is an interior point

    Returns:
        ndarray: measure mu computed at every vertex
    """
    mu_from_grid = np.zeros(_coordinates.shape[0])
    for i in range(_coordinates.shape[0]):
        if _boundaryMask[i]:
            continue  # skip boundary points
        neighbourhood = _adjacancies[i]
        next = np.array([1, 2, 0])
        for j in range(_countADJ[i]):
            triag = _triangles[neighbourhood[j][0]]
            # vertex0 is equivalent to i
            vertex1 = triag[next[neighbourhood[j][1]]]
            vertex2 = triag[next[next[neighbourhood[j][1]]]]

            # implementing second order quadrature from Saleri p. 415
            midpoint1 = np.array([(_coordinates[i]+_coordinates[vertex1])/2])
            midpoint2 = np.array([(_coordinates[i]+_coordinates[vertex2])/2])
            vec1 = _coordinates[vertex1]-_coordinates[i]
            vec2 = _coordinates[vertex2]-_coordinates[i]
            area = np.abs(vec1[0] * vec2[1] - vec1[1] * vec2[0])/2.0
            mu_from_grid[i] = mu_from_grid[i] + \
                (f(midpoint1)[0]+f(midpoint2)[0])*area/6.0
    return mu_from_grid


def computeMuFromPoints(_coordinates, _boundaryMask):
    """calculates a possible measure mu from an initial vertices by applying
    a delaunay triangulation and using "computeMuFromGrid"
    See chapter 3.2.3 in the thesis for further information.

    Args:
        _coordinates (ndarray): array [[x1,y1],[x2,y2],...] with coordinate data of
            the vertices
        _boundaryMask (ndarray): mask for _coordinates with 1 if the desired vertex
            is an boundary point and 0 if it is an interior point

    Returns:
        ndarray: measure mu computed at every vertex
    """
    tri = Delaunay(_coordinates)
    count, adj, _ = optri.findAdjacencies(
        np.copy(_coordinates), np.copy(tri.simplices))

    return computeMuFromGrid(adj, tri.simplices, count, _coordinates, _boundaryMask)


def exactSolution(val):
    """function u which holds the exact solution values of
    our Monge-Ampere-Equation.

    Args:
        val (ndarray): array [[x1,y1],[x2,y2],...] with data from the domain

    Returns:
        ndarray: u(x,y) for every (x,y) pair as a list
    """
    if FUNCTION == 1:
        # parabola
        return np.square(val[:, 0]) + np.square(val[:, 1])
    elif FUNCTION == 2:
        # tangens of the parbola
        return np.tan(np.square(val[:, 0]) + np.square(val[:, 1]))
    elif FUNCTION == 3:
        # abs function
        return np.abs(val[:, 0])+np.abs(val[:, 1])-1
    elif FUNCTION == 4:
        # exponential parabola
        return np.exp((np.square(val[:, 0]) + np.square(val[:, 1]))/2)
    elif FUNCTION == 5:
        # sin fraction
        return -(np.sin(val[:, 0])*np.sin(val[:, 1]))/(np.sin(val[:, 0])+np.sin(val[:, 1]))
    else:
        raise ValueError("function not implemented")


def create_square_points(n, beta):
    """creates the grid points for the square domain

    Args:
        n (int): number of grid points per dimension
        beta (float): scaling factor to generate a graduated grid

    Returns:
        coords (ndarray): list of x,y coordinates of the grid points
        bound (ndarray): mask for coords with 1 if the desired vertex is a boundary point
    """
    x = np.linspace(-1, 1, endpoint=True, num=n, dtype=np.float64)
    y = np.linspace(-1, 1, endpoint=True, num=n, dtype=np.float64)

    x = np.power(np.abs(x), beta)*np.sign(x)
    y = np.power(np.abs(y), beta)*np.sign(y)
    u, v = np.meshgrid(x, y)
    u, v = u.flatten(), v.flatten()

    coordinates = np.stack((u, v))
    coordinates = coordinates.transpose()

    # True when boundary, False when interior point
    boundaryMask = np.logical_or(np.logical_or(u == 1, u == -1),
                                 np.logical_or(v == 1, v == -1))

    if FUNCTION == 5:
        print("transforming coordinates to [0,pi]x[0,pi] for function 5")
        coordinates = coordinates + np.array([1, 1])
        coordinates = coordinates * np.pi/2

    return coordinates, boundaryMask


def create_circular_points(ppc, nrbc):
    """generates a mesh of the unit circle consisting of
    multiple circles with a given number of points per circle

    automatically adds a point at the origin
    Args:
        ppc (int): points per circle
        nrbc (int): number of circles

    Returns:
        coords (ndarray): list of x,y coordinates of the grid points
        bound (ndarray): mask for coords with 1 if the desired vertex is a boundary point
    """
    circle = np.linspace(0, 2*np.pi, ppc, endpoint=False)
    radii = np.linspace(1, 0, nrbc, endpoint=False)

    x = np.outer(radii, np.cos(circle)).flatten()
    y = np.outer(radii, np.sin(circle)).flatten()
    x = np.append(x, 0)
    y = np.append(y, 0)

    coordinates = np.stack((x, y)).T
    boundaryMask = np.append(np.ones(ppc, dtype=bool),
                             np.zeros(x.shape[0]-ppc, dtype=bool))

    return coordinates, boundaryMask


def planarity_check(points):
    """checks whether a given point cloud is planar

    Args:
        points (ndarray): 3d point cloud [[x1,y1,z1],[x2,y2,z2],...]

    Returns:
        bool: true if the point cloud is planar, false otherwise
    """
    p0 = points[0]
    p1 = points[1]
    p2 = points[-1]
    v1 = p1-p0
    v2 = p2-p0
    n = np.cross(v1, v2)
    n = n/np.linalg.norm(n)
    remaining_points = points[2:-1]
    remaining_points = remaining_points-p0
    dotp_vals = np.dot(remaining_points, n)
    return np.all(np.isclose(dotp_vals, 0.0))


def create_init_triangulation(path, coords, bound, plot=False):
    """Creates the initial function graph in the space W_M for the Oliker-Prussner-Method.
    This function is the lower boundary of the convex hull of the boundary points (the function
    values of the boundary points are given through the boundary condition of the Monge-
    Ampère-Equation and are implemented through oliker_prussner_init.exact_solution).
    See also chapter 3.2.2 in the thesis for further information.

    This function also calculates the measure mu for every vertex of the triangulation from the given
    right hand side function f (implemented by oliker_prussner_init.f). It uses the 
    method described in the thesis chapter 3.2.3 (see oliker_prussner_init.computeMuFromPoints
    for further information).

    Args:
        path (string): path to the folder where the data should be saved
        coords (ndarray): list of x,y coordinates of the grid points
        bound (ndarray): mask for coords with 1 if the desired vertex is a boundary point
        plot (bool, optional): whether to plot the initial triangulation and graph. Defaults to False.

    Returns:
        coords_sorted: the sorted coordinate list, starting with all boundary points.
            May be identical to input list, if already sorted.
        bound_sorted: the sorted boundary mask, starting with all boundary points.
            May be identical to input list, if already sorted.
    """
    # sort boundary points to the beginning of the point cloud
    coords = np.append(coords[bound], coords[~bound], axis=0)
    bound = np.append(bound[bound], bound[~bound], axis=0)
    val = exactSolution(coords)

    if FUNCTION == 5 and not circular_point_cloud:
        # repair the nan evaluation at the origin
        # by continous extension of the function (u(0,0)=0)
        val = np.nan_to_num(val)
        # smooth boundary values
        val[np.isclose(val, 0.0)] = 0.0
    # set boundary values to the exact solution
    hullP = np.vstack(
        (coords[bound].T, [val[bound]])).T

    indices = np.arange(coords.shape[0])
    if planarity_check(hullP):
        # 3d version of scipy convex hull (used in optri.getLowerConvexHull)
        # is not working with planar 3d point cloud
        hullP2d = coords[bound]
        delauney2d = Delaunay(hullP2d)
        # set the z coordinate of the unused points to 0 because
        # the z coordinate will be calculated in optri.refineTriangulation
        remainingIndices = np.setdiff1d(indices, delauney2d.simplices)
        points = np.vstack((coords.T, np.repeat(0, coordinates.shape[0]))).T
        points[delauney2d.simplices, 2] = val[delauney2d.simplices]
        # complete the convex hull of some/all boundary points to
        # a triangulation of the whole domain (every vertex is a grid point)
        points, triags = optri.refineTriangulation(
            delauney2d.simplices, points, remainingIndices)
    else:
        lowerHull = optri.getLowerConvexHull(hullP)
        # set the z coordinate of the unused points to 0 because
        # the z coordinate will be calculated in optri.refineTriangulation
        remainingIndices = np.setdiff1d(indices, lowerHull.vertices)
        points = np.vstack((coords.T, np.repeat(0, coordinates.shape[0]))).T
        points[lowerHull.vertices, 2] = val[lowerHull.vertices]
        # complete the convex hull of some/all boundary points to
        # a triangulation of the whole domain (every vertex is a grid point)
        points, triags = optri.refineTriangulation(
            lowerHull.simplices, points, remainingIndices)

    triags = optri.sortTrianglesCCW(
        np.copy(coords), np.copy(triags))

    if plot:
        plt.figure(10)
        plt.title("initial triangulation")
        plt.triplot(points[:, 0], points[:, 1], triags,
                    marker="o", color="green", linewidth=2)
        opplot.show(triags, points[:, :2],
                    points[:, 2], 11, "initial function")

    count, adj, elementIdx = optri.findAdjacencies(
        np.copy(coords), np.copy(triags))
    muPoints = computeMuFromPoints(coords, bound)

    np.savetxt(os.path.join(path, "coordinates.dat"), coords)
    np.savetxt(os.path.join(path, "elements.dat"), triags, fmt="%i")
    np.savetxt(os.path.join(path, "init.dat"), points[:, 2])
    np.savetxt(os.path.join(path, "boundaryMask.dat"), bound, fmt="%i")
    np.savetxt(os.path.join(path, "solution.dat"), val)
    np.savetxt(os.path.join(path, "muFromPoints.dat"), muPoints)
    np.savetxt(os.path.join(path, "elementIdx.dat"), elementIdx, fmt="%i")
    np.savetxt(os.path.join(path, "countADJ.dat"), count, fmt="%i")
    np.save(os.path.join(path, "adjacencies"), adj)

    return coords, bound


def create_solution_data(path, coords, bound, plot=False):
    """creates the function graph in the space W_M of the solution 
    function by taking the lower boundary of the convex hull of all function points
    (the function values are given by the exact solution implemented by oliker_prussner_init.exactSolution). 

    Also calculates the Monge-Ampère-Measure of the solution at every vertex of the triangulation 
    by the method described in chapter 3.3.1 of the thesis (see oliker_prussner_triangulation.calculateMongeAmpereMeasure
    for further information).

    Args:
        path (string): path to the folder where the data should be saved
        coords (ndarray): list of x,y coordinates of the grid points
        bound (ndarray): mask for coords with 1 if the desired vertex is a boundary point
        plot (bool, optional): whether to plot the initial triangulation and graph. Defaults to False.
    """
    bound = bound.astype(bool)
    val = exactSolution(coords)
    indices = np.arange(coords.shape[0])

    if FUNCTION == 5 and not circular_point_cloud:
        # repair the nan evaluation at the origin
        val = np.nan_to_num(val)
    hullP = np.vstack((coords.T, [val])).T
    lowerHull = optri.getLowerConvexHull(hullP)
    remainingIndices = np.setdiff1d(indices, lowerHull.vertices)
    _, triags = optri.refineTriangulation(
        lowerHull.simplices, lowerHull.points, remainingIndices)

    triags = optri.sortTrianglesCCW(np.copy(coords),
                                    np.copy(triags))
    if plot:
        plt.figure(199)
        plt.title("solution triangulation")
        plt.triplot(coords[:, 0], coords[:, 1], triags,
                    marker="o", color="green", linewidth=2)
    count, adj, elementIdx = optri.findAdjacencies(
        np.copy(coords), np.copy(triags))

    exactMu = np.zeros(coords.shape[0])
    for i in range(coords.shape[0]):
        if bound[i]:
            continue
        exactMu[i] = optri.calculateMongeAmpereMeasure(
            i, adj, triags, count, coords, val)
    np.savetxt(os.path.join(path, "solutionMu.dat"), exactMu)
    np.savetxt(os.path.join(path, "solutionTriags.dat"), triags, fmt="%i")

    if plot:
        plt.figure(200)
        plt.title("solution")
        opplot.show(lowerHull.simplices, coords, val, 200)


if __name__ == "__main__":
    if len(sys.argv) not in [5, 6]:
        print(
            "Usage: python oliker_prussner_init.py <\"circle\"|\"square\"> <param1> <param2> <function> [<plot=False>]")
        sys.exit(1)

    if sys.argv[1] == "circle":
        circular_point_cloud = True
    elif sys.argv[1] == "square":
        circular_point_cloud = False
    else:
        print("first argument must be either \"circle\" or \"square\"")
        sys.exit(1)

    param1 = int(sys.argv[2])
    if param1 < 3:
        print("param1 must be at least 3")
        sys.exit(1)

    if circular_point_cloud:
        param2 = int(sys.argv[3])
        if param2 < 1:
            print("param2 must be at least 1 for circular point cloud")
            sys.exit(1)
    else:
        param2 = float(sys.argv[3])
        if param2 < 0.0 or np.isclose(param2, 0.0) or param2 > 1.0:
            print("param2 must be in (0,1] for square point cloud")
            sys.exit(1)

    FUNCTION = int(sys.argv[4])
    if FUNCTION < 1 or FUNCTION > 5:
        print("Function argument must be between 1 and 5")
        sys.exit(1)

    plot = False
    if len(sys.argv) == 6:
        if sys.argv[5] == "True" or sys.argv[5] == "true":
            plot = True

    opplot.bigprint("Oliker-Prussner Method: Initialisation")
    if circular_point_cloud:
        print(
            f"create circular point cloud with {param1} points per circle and {param2} circles")
        print(
            f"create initial function with predefined function {FUNCTION}")
        folder_name = f"circle_{param1}_{param2}_{FUNCTION}"
        if not os.path.isdir(folder_name):
            print("create folder: " + folder_name)
            os.mkdir(folder_name)
        coordinates, boundaryMask = create_circular_points(
            param1, param2)
    else:
        print(
            f"create square point cloud with {param1} points per dimension and scaling factor {param2}")
        print(
            f"create initial function with predefined function {FUNCTION}")
        folder_name = f"square_{param1}_{param2}_{FUNCTION}"
        if not os.path.isdir(folder_name):
            print("create folder: " + folder_name)
            os.mkdir(folder_name)
        coordinates, boundaryMask = create_square_points(
            param1, param2)

    coordinates_sorted, boundaryMask_sorted = create_init_triangulation(
        folder_name, coordinates, boundaryMask, plot)
    create_solution_data(
        folder_name, coordinates_sorted, boundaryMask_sorted, plot)

    if plot:
        plt.show()
    print("initialisation done")
