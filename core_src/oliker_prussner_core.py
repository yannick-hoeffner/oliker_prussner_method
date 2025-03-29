######################################################################################################
#
#   implementation of the oliker-prussner-method
#
#   by Yannick Höffner, Friedrich-Schiller-University Jena,
#   Bachelor Thesis WS 2024/25
#
#   based on the original paper from V.I.Oliker and L.D.Prussner:
#   "On the Numerical Solution of the Equation [...] and Its Discretizations, I"
#   Numerische Mathematik 54.3 (1989): 271-294
#
#   This file contains a minimal version of the Oliker-Prussner-Method, that calculates a final solution
#   given just set of grid points, boundary conditions and a given measure mu.
#
######################################################################################################
import numpy as np
import numpy.typing as npt
from numba import jit

import core_src.oliker_prussner_coretriag as optri

########################################################################################################
#   IMPORTANT EQUATION CALCULATIONS
#   (see Appendix B in the thesis)
########################################################################################################


@jit(nopython=True)
def calculate_alpha(index: np.int64, coordinates: npt.NDArray[np.float64],
                    adjNext: npt.NDArray[np.int64], uValuesAdj: npt.NDArray[np.float64]) -> tuple[
                        npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """This function calculates the alpha values for the neighborhood of a vertex.
    See chapter 3.3.2 and appendix B.2 in the thesis for more information.

    Args:
        index (int64): the index under consideration
        coordinates (NDArray[float64]): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        adjNext (NDArray[int64]): result of optri.getNeighborhoodCCW
        uValuesAdj (NDArray[float64]): current function values at the adjacent vertices

    Returns:
        alphas (NDArray[float64]): array of alpha values for the neighborhood
        angles (NDArray[float64]): array of outer angles for the neighborhood
    """
    alphas = np.zeros(adjNext.size-2, dtype=np.float64)
    angles = np.zeros(adjNext.size-2, dtype=np.float64)
    for i in range(1, adjNext.size-1):
        A_ij = np.append(coordinates[adjNext[i-1]], uValuesAdj[i-1])
        A_ij1 = np.append(coordinates[adjNext[i]], uValuesAdj[i])
        A_ij2 = np.append(coordinates[adjNext[i+1]], uValuesAdj[i+1])
        # set A_i[z] artificially to 0.0 to use cross products of p_i without z-influence
        A_i = np.append(coordinates[index], [0.0])

        Nbar_ij = np.cross(A_ij-A_i, A_ij1-A_i)
        Nbar_ij1 = np.cross(A_ij1-A_i, A_ij2-A_i)

        # calculating alpha for solution range
        # notation not from the paper
        D_alpha_j = np.array([A_ij1[1]-A_ij[1], A_ij[0]-A_ij1[0]])
        D_alpha_j1 = np.array([A_ij2[1]-A_ij1[1], A_ij1[0]-A_ij2[0]])

        a_alpha = Nbar_ij[:2]*Nbar_ij1[2] - Nbar_ij1[:2]*Nbar_ij[2]
        b_alpha = D_alpha_j1 * Nbar_ij[2] - D_alpha_j * Nbar_ij1[2]

        # sometimes one of the two components is zero in both a and b;
        # this depends on the position of the vertices
        if np.abs(b_alpha[1]) > 0.000001:
            alphas[i-1] = a_alpha[1]/b_alpha[1]
        elif np.abs(b_alpha[0]) > 0.000001:
            alphas[i-1] = a_alpha[0]/b_alpha[0]
        else:
            # this vertex lies on the line of the two adjacent vertices;
            # this vertex will never change the subgradient
            # since this vertex will always be on the face spanned by
            # Ai and the two adjacent vertices
            alphas[i-1] = np.inf

        # calculate outer angle of this point in the neigbourhood
        v_jj1 = A_ij - A_ij1
        v_jj2 = A_ij2 - A_ij1
        dot = v_jj1[0]*v_jj2[0] + v_jj1[1]*v_jj2[1]
        det = v_jj1[0]*v_jj2[1] - v_jj1[1]*v_jj2[0]
        angles[i-1] = np.degrees(np.arctan2(det, dot) % (2*np.pi))

    return alphas, angles


@jit(nopython=True)
def build_equation_parameters(index: np.int64, coordinates: npt.NDArray[np.float64],
                              adjNext: npt.NDArray[np.int64], uValuesAdj: npt.NDArray[np.float64]) -> tuple[
                                  np.float64, np.float64, np.float64]:
    """calculates the equation parameters a,b,c for the quadratic equation
    that represents the calculation of the Monge-Ampère-Measure for a vertex
    with unknown z-value.
    See chapter 3.3.2 and appendix B.1 in the thesis for more information.

    Args:
        index (int64): the index under consideration
        coordinates (NDArray[float64]): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        adjNext (NDArray[int64]): result of optri.getNeighborhoodCCW
        uValuesAdj (NDArray[float64]): current function values at the adjacent vertices

    Returns:
        a,b,c (float64): the parameters a,b,c from the quadratic equation
    """
    a, b, c = 0, 0, 0
    for i in range(1, adjNext.size-1):
        # for detailed information about definition and naming of
        # variables in this loop, see the original Olliker Prussner Paper
        # on page 288 in the function new_z

        A_ij = np.append(coordinates[adjNext[i-1]], uValuesAdj[i-1])
        A_ij1 = np.append(coordinates[adjNext[i]], uValuesAdj[i])
        A_ij2 = np.append(coordinates[adjNext[i+1]], uValuesAdj[i+1])
        # set A_i[z] artificially to 0.0 to use cross products of p_i without z-influence
        A_i = np.append(coordinates[index], [0.0])

        Nbar_ij = np.cross(A_ij-A_i, A_ij1-A_i)
        Nbar_ij1 = np.cross(A_ij1-A_i, A_ij2-A_i)
        gamma_ij = Nbar_ij[2]
        gamma_ij1 = Nbar_ij1[2]

        N_ij = Nbar_ij/gamma_ij
        N_ij1 = Nbar_ij1/gamma_ij1

        D_ij = np.array([A_ij1[1]-A_ij[1], A_ij[0]-A_ij1[0]])/gamma_ij
        D_ij1 = np.array([A_ij2[1]-A_ij1[1], A_ij1[0]-A_ij2[0]])/gamma_ij1

        a_val = D_ij[0] * D_ij1[1] - D_ij[1] * D_ij1[0]
        b_val = D_ij[0] * N_ij1[1] - D_ij[1] * N_ij1[0]
        b_val = b_val + N_ij[0] * D_ij1[1] - N_ij[1] * D_ij1[0]
        c_val = N_ij[0] * N_ij1[1] - N_ij[1] * N_ij1[0]

        a = a + a_val
        b = b + b_val
        c = c + c_val

    a = -a/2
    b = -b/2
    c = -c/2
    return a, b, c


########################################################################################################
#   CORE FUNCTIONALITY OF THE OLIKER-PRUSSNER-METHOD
#   (see chapter 3 in the thesis)
########################################################################################################


def olliker_prussner_step(index: np.int64, coordinates: npt.NDArray[np.float64],
                          triangles: npt.NDArray[np.int64], adjacancies: npt.NDArray,
                          count_adj: npt.NDArray[np.int64], old_z_values: npt.NDArray[np.float64],
                          measure: npt.NDArray[np.float64]) -> np.float64:
    """This function implements one z-update step of the Oliker-Prussner-Method for a single vertex.
    This function basically implements algorithm 3 of the thesis.
    See chapter 3.3.2 in the thesis for more information.

    Args:
        index (int64): the index under consideration
        coordinates (NDArray[float64]): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        triangles (NDArray[int64]): see optri
        adjacancies (NDArray[any]): see optri
        count_adj (NDArray[int64]): see optri
        old_z_values (NDArray[float64]): array of current function values at the vertices
        measure (NDArray[float64]): array of the pointwise measure mu, that we want to achieve (for every vertex)

    Returns:
        z (float64): updated z-value for the vertex
    """
    current_value = old_z_values[index]
    adjNext, _ = optri.get_neighborhood_CCW(
        index, triangles, adjacancies,  count_adj)
    while True:
        uValuesAdj = old_z_values[adjNext]
        ###########################################################
        # calculate solution range (alpha)
        ###########################################################
        alphas, angles = calculate_alpha(
            index, coordinates, adjNext, uValuesAdj)
        max_alpha_idx = -1
        max_alpha = -np.inf
        for i in range(alphas.size):
            # only if the outer angle is less than 180°,
            # then this vertex will not be on the convex hull
            # if you go down deeper in the z direction
            if angles[i] < 180:
                if alphas[i] > max_alpha and alphas[i] < current_value + 0.000001:
                    max_alpha = alphas[i]
                    max_alpha_idx = i

        ###########################################################
        # build sub gradient equation
        ###########################################################
        # coefficients for the quadratic equation az^2+bz+c
        a, b, c = build_equation_parameters(
            index, coordinates, adjNext, uValuesAdj)
        ###########################################################
        # solve sub gradient equation
        ###########################################################
        # equation -mu_i = a * z^2 + b * z + c

        c = c + measure[index]

        # equation  0 = a * z^2 + b * z + c
        b2M4ac = b*b-4*a*c
        sol: list[np.float64] = []
        if (np.abs(a) < 0.00001):
            sol.extend([-c/b])
            print("a param near zero")
        else:
            sol.extend([(-b + np.sqrt(b2M4ac))/(2*a),
                        (-b - np.sqrt(b2M4ac))/(2*a)])

        sol = np.array(sol)
        sol2 = sol[np.where(sol < current_value)]
        sol2 = sol2[np.where(sol2 > max_alpha)]
        if len(sol2) > 0:
            return np.max(sol2)
        if max_alpha == -np.inf:
            # i currently don't know why this case happens from time to time,
            # but we just skip the vertex if it happens and let the "self healing"
            # property of global operation of the Oliker-Prussner-Method do its job
            print("huh")
            return current_value
        # remove the vertex, that will not be on the convex hull
        # and repeat the calculation
        if max_alpha_idx == 0:
            adjNext = np.delete(adjNext, adjNext.size-1)
            adjNext = np.delete(adjNext, 1)
            adjNext = np.append(adjNext, adjNext[1])
        elif max_alpha_idx == alphas.size-1:
            adjNext = np.delete(adjNext, adjNext.size-2)
            adjNext = np.delete(adjNext, 0)
            adjNext = np.append(adjNext, adjNext[1])
        else:
            adjNext = np.delete(adjNext, max_alpha_idx+1)


def oliker_prussner_sceme(coordinates: npt.NDArray[np.float64], boundary_mask: npt.NDArray[np.bool_],
                          initial_triags: npt.NDArray[np.int64], initial_values: npt.NDArray[np.float64],
                          measure: npt.NDArray[np.float64], infostring: str, max_iterations: np.int64) -> tuple[
                              npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], np.int64]:
    """This function implements the Oliker-Prussner-Method.
    See chapter 3 in the thesis or the original paper from Oliker and Prussner for more information.

    Args:
        coordinates (NDArray[float64]): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        boundary_mask (NDArray[bool]): boundary mask of the coordinate list
        initial_triags (NDArray[int64]): triangle array of start function; see optri
        initial_values (NDArray[float64]): function values of the start function at the vertices
        measure (NDArray[float64]): the Monge-Ampère-Measure that we want to achieve
        infostring (string): string that will be displayed in the console
        max_iterations (int64): maximum number of iterations

    Returns:
        calculated_z_values (NDArray[float64]): updated function values
        calculated_triags (NDArray[int64]): updated triangulation array
        nbr_iterations (int64): number of iterations needed
    """
    np.set_printoptions(precision=16)
    monge_ampere_precision = 1e-6
    nbr_iterations = 0

    # calculate adjacencie information from the triangulation
    count_adj, adjacencies, _ = optri.find_adjacencies(
        coordinates, initial_triags)
    # rename the variables for better readability
    calculated_z_values = initial_values
    calculated_triags = initial_triags

    for repetitions in range(max_iterations):
        innerIndices = np.arange(coordinates.shape[0])[~boundary_mask]
        new_z = np.copy(calculated_z_values)

        ###########################################################
        # adjust z values by Monge-Ampère-Measure calculation
        ###########################################################

        calculatedIndices = 0
        for index in innerIndices:
            monge_ampere_measure = optri.calculate_monge_ampere_measure(
                index, coordinates, calculated_triags, adjacencies, count_adj, calculated_z_values)
            if np.abs(monge_ampere_measure-measure[index]) < monge_ampere_precision:
                # the curvature/bulge of the function is already correct at this vertex
                continue
            new_z[index] = olliker_prussner_step(index, coordinates, calculated_triags, adjacencies,
                                                 count_adj, calculated_z_values, measure)
            calculatedIndices = calculatedIndices + 1
        if calculatedIndices == 0:
            # break iteration if Monge-Ampère-Measure is equal to the given measure mu (up to tollerance)
            # for all inner vertices
            print("no more update done, iteration finished")
            nbr_iterations = repetitions
            break

        ###########################################################
        # globally overrite the z values
        ###########################################################
        calculated_z_values = new_z

        ###########################################################
        # recalculate the triangulation with the new z values
        ###########################################################

        hullP = np.vstack((coordinates.T, [calculated_z_values])).T
        hull = optri.get_lower_convex_hull(hullP)
        print("lower convex hull calculated")

        remainingIndices = np.setdiff1d(innerIndices, hull.vertices)
        print("remaining points: ", remainingIndices)

        points, triangles = optri.refine_triangulation(
            hull.simplices, hull.points, remainingIndices)
        calculated_z_values = np.copy(points[:, 2])

        calculated_triags = optri.sort_triangles_CCW(
            coordinates, np.copy(triangles))
        count_adj, adjacencies, _ = optri.find_adjacencies(
            np.copy(coordinates), calculated_triags)
        ###########################################################
        # print iteration information
        ###########################################################
        print(f"{infostring} it:{repetitions}")

    return calculated_z_values, calculated_triags, nbr_iterations
