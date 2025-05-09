#####################################################################################################
#
#   functions related to the management of the triangulation for the oliker-prussner-method,
#   not all functions are actually used, some are for testing/debugging
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
from numba import jit
from scipy.spatial import ConvexHull, Delaunay


@jit(nopython=True)
def intersect_line_triangle(point: npt.NDArray[np.float64], p1: npt.NDArray[np.float64],
                            p2: npt.NDArray[np.float64], p3: npt.NDArray[np.float64]) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """calculates the intersection of a vertical line
    (directional vector [0,0,1]) starting at 'point' with
    a triangle in R^3 build by the vertices 'p1','p2','p3'

    Args:
        point (NDArray[float64]): x, y coordinates of the starting point of the
            ray, z is set to 0
        p1 (NDArray[float64]): x,y,z coordinates of first triangle vertex
        p2 (NDArray[float64]): x,y,z coordinates of second triangle vertex
        p3 (NDArray[float64]): x,y,z coordinates of third triangle vertex

    Returns:
        N (NDArray[float64]): the normal vector of the triangle (R^3), None if N[2] = 0
        p (NDArray[float64]): the intersection point of the line and triangle, None if
            intersection does not exist or N[2] = 0
        dist (NDArray[float64]): the signed distances of the plane intersection point to the
            different triangle sides, None if N[2] = 0
    """
    # TODO: maybe adjust the algorithm
    # we are basically performing lots of ray/triangle intersections
    # wich can be done better with the  Möller and Trumbore algorithm
    #
    # there also exist (seemingly) even better algorithms:
    # Doug Baldwin and Michael Weber, Fast Ray-Triangle Intersections by Coordinate Transformation
    # https://jcgt.org/published/0005/03/03/

    # current code from stackoverflow
    # https://stackoverflow.com/questions/54143142/3d-intersection-between-segment-and-triangle
    # by user: Mauricio Cele Lopez Belon

    # triangle normal vector
    N = np.cross((p2 - p1), (p3 - p1))

    if abs(N[2]) < 0.000001:
        return None, None, None
    # intersection point of triangle plane and line
    p = point - (-np.dot(N, p1) + np.dot(N, point)) / \
        N[2] * np.array([0, 0, 1])
    # explanaition: N[2] == np.dot(N, [0,0,1])

    # normal vectors of triangle sides within the triangle plane
    N12 = np.cross((p2 - p1), N)
    N23 = np.cross((p3 - p2), N)
    N31 = np.cross((p1 - p3), N)

    # distance of intersection point to triangle sides
    Dist1 = np.dot((p - p1), N12) / np.linalg.norm(N12)
    Dist2 = np.dot((p - p2), N23) / np.linalg.norm(N23)
    Dist3 = np.dot((p - p3), N31) / np.linalg.norm(N31)
    if Dist1 < 0.00001 and Dist2 < 0.00001 and Dist3 < 0.00001:
        return N, p, np.array((Dist1, Dist2, Dist3))
    return N, None, np.array((Dist1, Dist2, Dist3))


def sort_triangles_CCW(coordinates: npt.NDArray[np.float64], triags: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """sorts the vertices of the given triangles, such that
    they are in ccw order

    Args:
        coordinates (NDArray[float64]): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        triags (NDArray[int64]): array of triangle vertex indices [ [t1v1, t1v2, t1v3], [t2v1, t2v2, t2v3], ...]

    Returns:
        triags (NDArray[int64]): the sorted triangle array
    """
    for i in range(triags.shape[0]):
        triangle = triags[i]
        coords = coordinates[triangle]
        center = np.mean(coords, axis=0)
        localCoords = coords - center
        arg = np.arctan2(localCoords[:, 1], localCoords[:, 0])
        idx_CCW = np.argsort(arg)
        triags[i] = triags[i][idx_CCW]
    return triags


def find_adjacencies(coordinates: npt.NDArray[np.float64], triags: npt.NDArray[np.int64]) -> tuple[
        npt.NDArray[np.int64], npt.NDArray, npt.NDArray[np.int64]]:
    """finds the adjacent triangle of the vertices and stores
    the indices in a linked list

    the adjacent triangles inside the linked list are stored in
    cw order and ccw order.
    FOR PROPPER SORTING, TRIANGLE VERTICES MUST BE IN CCW ORDER

    Args:
        coordinates (NDArray[float64]): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        triags (NDArray[int64]): array of triangle vertex indices [ [t1v1, t1v2, t1v3], [t2v1, t2v2, t2v3], ...]

    Returns:
        count (NDArray[int64]): array where count[i] is the number of adjacent triangles to vertex i
        adj (NDArray[any]): array where adj[i] is the linked list of adjacent triangles to vertex i.
            The k-th entry in this linked list is:\n
            adj[i][k][0] is the triangle index;\n
            adj[i][k][1] is the index of vertex i in triangle adj[i][0];\n
            adj[i][k][2] is the index in adj[i] where the cw next triangle appears;\n
            adj[i][k][3] is the index in adj[i] where the ccw next triangle appears;
        elementIdx (NDArray[int64]): array where elementIdx[i] contains the indices of the adjacencie-list
            of the vertices of the i-th triangle
    """
    maxNumberOfAdjacencies = 500
    # TODO: adjust the maximum number of vertices when domain changes
    adj = np.zeros(
        (coordinates.shape[0], maxNumberOfAdjacencies, 4), dtype=np.int32)
    # stores the number of adj vertices
    count = np.zeros((coordinates.shape[0]), dtype=np.int32)
    # stores the location of the triangles in the adj-array
    elementIdx = np.zeros(triags.shape)

    for i in range(triags.shape[0]):
        for j in range(3):
            k = triags[i][j]
            # for the k-th vertex the count[k]-th adjacent triangle is the i-th triangle
            adj[k][count[k]][0] = i
            # for the k-th vertex and the count[k]-th adjacent triangle,
            # the vertex k is stored as j-th edge of the triangle
            adj[k][count[k]][1] = j
            # store element idx
            elementIdx[i, j] = count[k]
            # increment the number of adjacend triangles
            count[k] = count[k]+1
    # currently the adjacent triangles are sorted by there appearence in elements.
    # now we need to sort the adjacent triangles, such that we can iterate through them
    # in clockwise order.
    next = np.array([1, 2, 0])
    for i in range(coordinates.shape[0]):
        # i = 7
        cnt = np.arange(count[i])
        # get list of all adjacent vertices
        adjInd = triags[adj[i, cnt, 0], next[adj[i, cnt, 1]]]
        # get relative angular coordinates
        localCoords = coordinates[adjInd] - coordinates[i]
        arg = np.arctan2(localCoords[:, 1], localCoords[:, 0])
        # sort angle descending
        idx_CW = np.flip(np.argsort(arg))
        idx_CCW = np.argsort(arg)
        idx_roll_CW = np.roll(idx_CW, +1)
        idx_roll_CCW = np.roll(idx_CCW, +1)
        temp_CW = np.zeros((count[i]))
        temp_CCW = np.zeros((count[i]))
        temp_CW[idx_roll_CW] = idx_CW
        temp_CCW[idx_roll_CCW] = idx_CCW
        # store index to next CW triangle
        adj[i, cnt, 2] = temp_CW
        # store index to next CCW triangle
        adj[i, cnt, 3] = temp_CCW
        # break
    return count, adj, elementIdx


def get_neighborhood_CW(index: np.int64, triangles: npt.NDArray[np.int64],
                        adjacencies: npt.NDArray, countADJ: npt.NDArray[np.int64]) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """reads out the adjacencies of a triangle mesh and returns a copy
    of the neighboring triangles and vertices in cw order, of the vertex 'index'

    Args:
        index (int64): the vertices-index where the neighborhood should be calculated
        triangles (NDArray[int64]): triangle description of function graph
        adjacancies (NDArray[any]): adjacencies relationship of the triangulation
        countADJ (NDArray[int64]): number of adjacent triangles for each vertex

    Returns:
        adjNext (NDArray[int64]): list of adjacent vertices in cw order
        triags (NDArray[int64]): list of adjacent triangles in cw order
    """
    CWsorting = adjacencies[index, :countADJ[index], 2]
    temp = np.zeros(countADJ[index], dtype=np.int32)
    for i in range(1, countADJ[index]):
        temp[i] = CWsorting[temp[i-1]]
    CWsorting = temp
    adj = adjacencies[index, :countADJ[index], :]
    triags = triangles[adj[:, 0]]
    next = np.array([1, 2, 0])
    adjNext = triags[np.arange(countADJ[index]), next[adj[:, 1]]]

    triags = triags[CWsorting]
    adjNext = adjNext[CWsorting]

    adjNext = np.append(adjNext, adjNext[0])
    adjNext = np.append(adjNext, adjNext[1])

    return np.copy(adjNext), np.copy(triags)


def get_neighborhood_CCW(index: np.int64, triangles: npt.NDArray[np.int64],
                         adjacencies: npt.NDArray, countADJ: npt.NDArray[np.int64]) -> tuple[
                             npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """reads out the adjacencies of a triangle mesh and returns a copy
    of the neighboring triangles and vertices in ccw order, of the vertex 'index'

    Args:
        index (int64): the vertices-index where the neighborhood should be calculated
        triangles (NDArray[int64]): triangle description of function graph
        adjacancies (NDArray[any]): adjacencies relationship of the triangulation
        countADJ (NDArray[int64]): number of adjacent triangles for each vertex

    Returns:
        adjNext (NDArray[int64]): list of adjacent vertices in ccw order
        triags (NDArray[int64]): list of adjacent triangles in ccw order
    """
    CCWsorting = adjacencies[index, :countADJ[index], 3]
    temp = np.zeros(countADJ[index], dtype=np.int32)
    for i in range(1, countADJ[index]):
        temp[i] = CCWsorting[temp[i-1]]
    CCWsorting = temp
    adj = adjacencies[index, :countADJ[index], :]
    triags = triangles[adj[:, 0]]
    next = np.array([1, 2, 0])
    adjNext = triags[np.arange(countADJ[index]), next[adj[:, 1]]]

    triags = triags[CCWsorting]
    adjNext = adjNext[CCWsorting]

    adjNext = np.append(adjNext, adjNext[0])
    adjNext = np.append(adjNext, adjNext[1])

    return np.copy(adjNext), np.copy(triags)


def calculate_monge_ampere_measure(index: np.int64, coordinates: npt.NDArray[np.float64],
                                   triangles: npt.NDArray[np.int64], adjacencies: npt.NDArray,
                                   count_adj: npt.NDArray[np.int64],  z_values: npt.NDArray[np.float64]) -> np.float64:
    """calculated the Monge-Ampère-Measure of the graph at vertex "index".
    see 'oliker_prussner_init' for more details on the input values

    Args:
        index (int64): the vertices-index where the Monge-Ampère-Measure should be calculated
        coordinates (NDArray[float64]): x and y coordinates of the vetices
        triangles (NDArray[int64]): triangle description of function graph
        adjacencies (NDArray[any]): adjacencies relationship of the triangulation
        count_adj (NDArray[int64]): number of adjacent triangles for each vertex
        z_values (NDArray[float64]): z coordinates of vertices

    Returns:
        monge_ampere_measure (float64): Monge-Ampère-Measure of vertex 'index'
    """
    adjNext, _ = get_neighborhood_CCW(
        index, triangles, adjacencies, count_adj)

    uValuesAdj = z_values[adjNext]
    coordsAdj = coordinates[adjNext]

    monge_ampere_measure = 0
    for k in range(1, adjNext.size-1):

        p_ij = np.append(coordsAdj[k-1], uValuesAdj[k-1])
        p_ij1 = np.append(coordsAdj[k], uValuesAdj[k])
        p_ij2 = np.append(coordsAdj[k+1], uValuesAdj[k+1])
        p_i = np.append(coordinates[index], z_values[index])

        Nbar_ij = np.cross(p_ij-p_i, p_ij1-p_i)
        Nbar_ij1 = np.cross(p_ij1-p_i, p_ij2-p_i)
        N_ij = np.array([Nbar_ij[0], Nbar_ij[1]])/Nbar_ij[2]
        N_ij1 = np.array([Nbar_ij1[0], Nbar_ij1[1]])/Nbar_ij1[2]

        monge_ampere_measure = monge_ampere_measure + \
            N_ij[0] * N_ij1[1] - N_ij[1] * N_ij1[0]

    monge_ampere_measure = monge_ampere_measure/2
    return np.abs(monge_ampere_measure)


@jit(forceobj=True)
def get_lower_convex_hull(hullPoints: npt.NDArray[np.float64]) -> ConvexHull:
    """calculated the convex hull of a given point cloud in R^3 and
    removes all triangles that do not belong to the lower part of the
    convex hull

    Args:
        hullPoints (NDArray[float64]): x, y, z coordinates of the points from which the hull
            should be build

    Returns:
        hull (scipy.spatial.ConvexHull): the convex hull object with modified simplices
    """
    hull = ConvexHull(hullPoints)  # , qhull_options="Q0")
    trianglePoints = hullPoints[hull.simplices]

    point = np.array([0.0, 0.0, 0.0])
    # remove vertical triangles
    for j in range(trianglePoints.shape[0]-1, -1, -1):
        triangle = trianglePoints[j]
        normalVec, intersect, _ = intersect_line_triangle(
            point, triangle[0], triangle[1], triangle[2])
        if normalVec is None:
            trianglePoints = np.delete(
                trianglePoints, j, 0)
            hull.simplices = np.delete(
                hull.simplices, j, 0)

    # remove upper triangles
    i = trianglePoints.shape[0]-1
    while i > -1:
        triangleI = trianglePoints[i]
        point = np.mean(triangleI, axis=0)
        _, intersectI, _ = intersect_line_triangle(
            point, triangleI[0], triangleI[1], triangleI[2])
        for j in range(trianglePoints.shape[0]-1, -1, -1):
            if i == j:
                continue
            triangleJ = trianglePoints[j]
            _, intersectJ, _ = intersect_line_triangle(
                point, triangleJ[0], triangleJ[1], triangleJ[2])
            if intersectJ is not None:
                if intersectJ[2] > intersectI[2] and np.abs(intersectJ[2]-intersectI[2]) > 1e-6:
                    trianglePoints = np.delete(
                        trianglePoints, j, 0)
                    hull.simplices = np.delete(
                        hull.simplices, j, 0)
                    if j < i:
                        i = i - 1
        i = i-1
    return hull


def find_points_on_lower_convex_hull(hullPoints: npt.NDArray[np.float64],
                                     projPoints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """generates a convex Hull from the points 'hullPoints' and
    calculates the projection of the points 'projPoints' onto the
    lower boundary of the convex hull (in z-direction)

    technically perfoming a ray tracing from a point in 'projPoints'
    by direction vector [0,0,1] and calculating intersections with the
    triangles that form the convex hull.

    DOES NOT CHECK, WHETHER AN INTERSECTION WITH THE CONVEX HULL EXISTS!

    Args:
        hullPoints (NDArray[float64]): array [[x1, y1, z1], ...] of 3D points to form the
            convex hull from
        projPoints (NDArray[float64]): array [[x1, y1], ...] of 2D points to be projected
            onto the convex hull

    Returns:
        NDArray[float64]: array [z1, z2, ...] of z-values to the corresponding 2D input points
    """
    hull = ConvexHull(hullPoints)
    trianglePoints = hullPoints[hull.simplices]

    finalZValues = np.zeros((projPoints.shape[0]))
    for i in range(projPoints.shape[0]):
        point = projPoints[i]
        finalPoint = np.array([point[0], point[1], 0.1e100])

        for j in range(trianglePoints.shape[0]-1, -1, -1):
            triangle = trianglePoints[j]
            _, intersect, _ = intersect_line_triangle(
                np.append(point, [0]), triangle[0], triangle[1], triangle[2])
            if intersect is not None:
                if intersect[2] < finalPoint[2]:
                    finalPoint = intersect
        finalZValues[i] = finalPoint[2]

    return finalZValues


@jit(forceobj=True)
def refine_triangulation(hullTriangles: npt.NDArray[np.int64], hullPoints: npt.NDArray[np.float64],
                         remainingIndices: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.float64],
                                                                           npt.NDArray[np.int64]]:
    """adds the remaining points to the triangulation of the convex hull
    by calculating the intersection of the points with the triangles of the
    convex hull. The new points are added to the hullPoints array and the
    new triangles are added to the hullTriangles array.

    Args:
        hullTriangles (NDArray[int64]): list of all triangles of the lower convex hull 
        hullPoints (NDArray[float64]): list of all 3d-points used to calculate the convex hull
            (must also contain the points that are currently not vertices of the convex hull!)
        remainingIndices (NDArray[int64]): list of indices of the points that are not part of the convex hull

    Returns:
        points (NDArray[float64]): points array with updated z-data
        triags (NDArray[int64]): triangle array with updated triangles
    """
    points = np.copy(hullPoints)
    triags = np.array([[0, 0, 0]])
    for i in range(hullTriangles.shape[0]):
        pi1, pi2, pi3 = hullTriangles[i]
        p1, p2, p3 = hullPoints[[pi1, pi2, pi3]]
        found_points = [p1, p2, p3]
        found_indices = [pi1, pi2, pi3]
        for j in remainingIndices:
            point = points[j]
            _, p, _ = intersect_line_triangle(
                np.array([point[0], point[1], 0]), p1, p2, p3)
            if p is not None:
                points[j, 2] = p[2]
                found_points.append(np.array([point[0], point[1], p[2]]))
                found_indices.append(j)
        found_points = np.array(found_points)
        found_indices = np.array(found_indices)

        tri = Delaunay(found_points[:, :2])
        # indices of tri are with respect to the enumeration of found_points
        # but we need the indices with respect to the enumeration of hullPoints
        tri_correctec = found_indices[tri.simplices]
        # scan triangulation for degenerated triangles
        for k in range(tri_correctec.shape[0]-1, -1, -1):
            local_coords = points[tri_correctec[k]].copy()
            local_coords[:, 2] = 1
            if np.isclose(np.linalg.det(local_coords), 0.0):
                tri_correctec = np.delete(tri_correctec, k, 0)
        triags = np.append(triags, tri_correctec, axis=0)

    triags = np.delete(triags, 0, axis=0)
    return points, triags
