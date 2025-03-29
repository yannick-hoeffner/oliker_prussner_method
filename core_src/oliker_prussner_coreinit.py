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
import numpy.typing as npt
from typing import Callable
from scipy.spatial import Delaunay
import core_src.oliker_prussner_coretriag as optri


########################################################################################################
#   MEASURE MU INITIALIZATION
#   (see chapter 3.2.3 in the thesis)
########################################################################################################


def compute_MAM_from_grid(adjacancies: npt.NDArray, triangles: npt.NDArray[np.int64],
                          countADJ: npt.NDArray[np.int64], coordinates: npt.NDArray[np.float64],
                          boundaryMask: npt.NDArray[np.bool_], rhs_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """calculates a possible Monge-Ampère-Measure from an initial grid by setting the
    local measure of a vertex to the integral over the classical FEM hat functions
    on adjoint triangles

    Args:
        adjacancies (NDArray[Any]): adjacencies list of initial triangulation;
            see optri.findAdjacencies for further information
        triangles (NDArray[int64]): triangles list of initial triangulation;
            see optri.findAdjacencies for further information
        countADJ (NDArray[int64]): adjacencies count of initial triangulation;
            see optri.findAdjacencies for further information
        coordinates (NDArray[float64]): array [[x1,y1],[x2,y2],...] with coordinate data of
            the vertices
        boundaryMask (NDArray[float64]): mask for coordinates with 1 if the desired vertex
            is an boundary point and 0 if it is an interior point
        rhs_f (callable): function for the right hand side of the Monge-Ampère-Equation

    Returns:
        mu_from_grid (NDArray[float64]): measure Monge-Ampère-Measure computed at every vertex
    """
    mu_from_grid = np.zeros(coordinates.shape[0], dtype=np.float64)
    for i in range(coordinates.shape[0]):
        if boundaryMask[i]:
            continue  # skip boundary points
        neighbourhood = adjacancies[i]
        next = np.array([1, 2, 0])
        for j in range(countADJ[i]):
            triag = triangles[neighbourhood[j][0]]
            # vertex0 is equivalent to i
            vertex1 = triag[next[neighbourhood[j][1]]]
            vertex2 = triag[next[next[neighbourhood[j][1]]]]

            # implementing second order quadrature from Saleri p. 415
            midpoint1 = np.array([(coordinates[i]+coordinates[vertex1])/2])
            midpoint2 = np.array([(coordinates[i]+coordinates[vertex2])/2])
            vec1 = coordinates[vertex1]-coordinates[i]
            vec2 = coordinates[vertex2]-coordinates[i]
            area = np.abs(vec1[0] * vec2[1] - vec1[1] * vec2[0])/2.0
            mu_from_grid[i] = mu_from_grid[i] + \
                (rhs_f(midpoint1)[0]+rhs_f(midpoint2)[0])*area/6.0
    return mu_from_grid


def compute_MA_measure(coordinates: npt.NDArray[np.float64], boundaryMask: npt.NDArray[np.bool_],
                       rhs_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """calculates a possible Monge-Ampère-Measure from an initial vertices by applying
    a delaunay triangulation and using "compute_mu_from_grid"
    See chapter 3.2.3 in the thesis for further information.

    Args:
        coordinates (NDArray[float64]): array [[x1,y1],[x2,y2],...] with coordinate data of
            the vertices
        boundaryMask (NDArray[bool]): mask for coordinates with 1 if the desired vertex
            is an boundary point and 0 if it is an interior point
        rhs_f (callable): function for the right hand side of the Monge-Ampère-Equation

    Returns:
        mu (NDArray[float64]): Monge-Ampère-Measure computed at every vertex
    """
    tri = Delaunay(coordinates)
    count, adj, _ = optri.find_adjacencies(
        np.copy(coordinates), np.copy(tri.simplices))

    return compute_MAM_from_grid(adj, tri.simplices, count, coordinates, boundaryMask, rhs_f)


########################################################################################################
#   TRIANGULATION INITIALIZATION
#   (see chapter 3.2.2 in the thesis)
########################################################################################################


def planarity_check(points: npt.NDArray[np.float64]) -> np.bool_:
    """checks whether a given point cloud is planar

    Args:
        points (NDArray[float64]): 3d point cloud [[x1,y1,z1],[x2,y2,z2],...]

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


def create_init_triangulation(coords: npt.NDArray[np.float64], boundary_mask: npt.NDArray[np.bool_], boundary_values: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Creates the initial function graph in the space W_M for the Oliker-Prussner-Method.
    This function is the lower boundary of the convex hull of the boundary points (the function
    values of the boundary points are given through the boundary condition of the Monge-
    Ampère-Equation). See also chapter 3.2.2 in the thesis for further information.

    THE COORDINATES/BOUNDARY MASK MUST BE SORTED WITH BOUNDARY POINTS FIRST!

    Args:
        coords (NDArray[float64]): list of x,y coordinates of the grid points, 
            where the boundary points should be sorted to the beginning of the list
        boundary_mask (NDArray[bool]): mask for coords; valued 1 if the desired vertex is a boundary point, otherwise 0;
            boundary points should be sorted to the beginning of the list
        boundary_values (NDArray[float64]): function values at the boundary points (all interior points should be 0!)
            boundary_values should not contain nan values.

    Returns:
        triags: the initial triangulation of the domain

        init_u_values: the initial function values at the grid points
    """
    # FIXME: add this to the CLI version
    # if FUNCTION == 5 and not circular_point_cloud:
    #     # repair the nan evaluation at the origin
    #     # by continous extension of the function (u(0,0)=0)
    #     boundary_values = np.nan_to_num(boundary_values)
    #     # smooth boundary values
    #     boundary_values[np.isclose(boundary_values, 0.0)] = 0.0

    # combine the boundary points with their function values given by the boundary condition
    hull_points = np.vstack(
        (coords[boundary_mask].T, [boundary_values[boundary_mask]])).T

    indices = np.arange(coords.shape[0])
    if planarity_check(hull_points):
        # 3d version of scipy convex hull (used in optri.getLowerConvexHull)
        # is not working with planar 3d point cloud
        hull_points_2d = coords[boundary_mask]
        delauney2d = Delaunay(hull_points_2d)
        # set the z coordinate of the unused points to 0 because
        # the z coordinate will be calculated in optri.refineTriangulation
        remainingIndices = np.setdiff1d(indices, delauney2d.simplices)
        init_u_values = np.vstack(
            (coords.T, np.repeat(0, coords.shape[0]))).T
        init_u_values[delauney2d.simplices,
                      2] = boundary_values[delauney2d.simplices]
        # complete the convex hull of some/all boundary points to
        # a triangulation of the whole domain (every vertex is a grid point)
        init_u_values, triags = optri.refine_triangulation(
            delauney2d.simplices, init_u_values, remainingIndices)
    else:
        lowerHull = optri.get_lower_convex_hull(hull_points)
        # set the z coordinate of the unused points to 0 because
        # the z coordinate will be calculated in optri.refineTriangulation
        remainingIndices = np.setdiff1d(indices, lowerHull.vertices)
        init_u_values = np.vstack(
            (coords.T, np.repeat(0, coords.shape[0]))).T
        init_u_values[lowerHull.vertices,
                      2] = boundary_values[lowerHull.vertices]
        # complete the convex hull of some/all boundary points to
        # a triangulation of the whole domain (every vertex is a grid point)
        init_u_values, triags = optri.refine_triangulation(
            lowerHull.simplices, init_u_values, remainingIndices)

    triags = optri.sort_triangles_CCW(
        np.copy(coords), np.copy(triags))

    return triags, init_u_values[:, 2]
