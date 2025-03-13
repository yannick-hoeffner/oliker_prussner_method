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
######################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import sys

import oliker_prussner_plot as opplot
import oliker_prussner_triangulation as optri


@jit(nopython=True)
def calculate_alpha(_index: int, _coordinates: np.ndarray, _adjNext: np.ndarray, _uValuesAdj: np.ndarray):
    """This function calculates the alpha values for the neighborhood of a vertex.
    See chapter 3.3.2 and appendix B.2 in the thesis for more information.

    Args:
        _index (int): the index under consideration
        _coordinates (np.ndarray): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        adjNext (np.ndarray): result of optri.getNeighborhoodCCW
        uValuesAdj (np.ndarray): current function values at the adjacent vertices

    Returns:
        alphas (np.ndarray): array of alpha values for the neighborhood
        angles (np.ndarray): array of outer angles for the neighborhood
    """
    alphas = np.zeros(_adjNext.size-2, dtype=np.float64)
    angles = np.zeros(_adjNext.size-2, dtype=np.float64)
    for i in range(1, _adjNext.size-1):
        A_ij = np.append(_coordinates[_adjNext[i-1]], _uValuesAdj[i-1])
        A_ij1 = np.append(_coordinates[_adjNext[i]], _uValuesAdj[i])
        A_ij2 = np.append(_coordinates[_adjNext[i+1]], _uValuesAdj[i+1])
        # set A_i[z] artificially to 0.0 to use cross products of p_i without z-influence
        A_i = np.append(_coordinates[_index], [0.0])

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
def build_equation_parameters(_index: int, _coordinates: np.ndarray, _adjNext: np.ndarray, _uValuesAdj: np.ndarray):
    """calculates the equation parameters a,b,c for the quadratic equation
    that represents the calculation of the Monge-Ampère-Measure for a vertex
    with unknown z-value.
    See chapter 3.3.2 and appendix B.1 in the thesis for more information.

    Args:
        _index (int): the index under consideration
        _coordinates (np.ndarray): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        adjNext (np.ndarray): result of optri.getNeighborhoodCCW
        uValuesAdj (np.ndarray): current function values at the adjacent vertices

    Returns:
        a,b,c (float): the parameters a,b,c from the quadratic equation
    """
    a, b, c = 0, 0, 0
    for i in range(1, _adjNext.size-1):
        # for detailed information about definition and naming of
        # variables in this loop, see the original Olliker Prussner Paper
        # on page 288 in the function new_z

        A_ij = np.append(_coordinates[_adjNext[i-1]], _uValuesAdj[i-1])
        A_ij1 = np.append(_coordinates[_adjNext[i]], _uValuesAdj[i])
        A_ij2 = np.append(_coordinates[_adjNext[i+1]], _uValuesAdj[i+1])
        # set A_i[z] artificially to 0.0 to use cross products of p_i without z-influence
        A_i = np.append(_coordinates[_index], [0.0])

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


def ollikerPrussnerStep(_index, _adjacancies, _triangles, _countADJ, _triangleIdx, _coordinates, _u, _mu):
    """This function implements one z-update step of the Oliker-Prussner-Method for a single vertex.
    This function basically implements algorithm 3 of the thesis.
    See chapter 3.3.2 in the thesis for more information.

    Args:
        _index (int): the index under consideration
        _adjacancies (np.ndarray): see optri
        _triangles (np.ndarray): see optri
        _countADJ (np.ndarray): see optri
        _triangleIdx (np.ndarray): see optri
        _coordinates (np.ndarray): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        _u (np.ndarray): array of current function values at the vertices
        _mu (np.ndarray): array of the pointwise measure mu, that we want to achieve (for every vertex)

    Returns:
        z float: updated z-value for the vertex
    """
    current_value = _u[_index]
    adjNext, _ = optri.getNeighborhoodCCW(
        _index, _adjacancies, _triangles, _countADJ)
    while True:
        uValuesAdj = _u[adjNext]
        ###########################################################
        # calculate solution range (alpha)
        ###########################################################
        alphas, angles = calculate_alpha(
            _index, _coordinates, adjNext, uValuesAdj)
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
            _index, _coordinates, adjNext, uValuesAdj)
        ###########################################################
        # solve sub gradient equation
        ###########################################################
        # equation -mu_i = a * z^2 + b * z + c

        c = c + _mu[_index]

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


def oliker_prussner_sceme(_coords, _bound, _adj, _triags, _countADJ, _u, _triagIDX, _mu, _sol, infostring, max_iterations):
    """This function implements the Oliker-Prussner-Method.
    See chapter 3 in the thesis or the original paper from Oliker and Prussner for more information.

    Args:
        _coords (_type_): array of vertex coordinates [[x1, y1], [x2, y2], ...]
        _bound (_type_): boundary mask of the coordinate list
        _adj (_type_): adjacency array of start function; see optri
        _triags (_type_): triangle array of start function; see optri
        _countADJ (_type_): adjacency counter array of start function; see optri
        _u (_type_): function values of the start function at the vertices
        _triagIDX (_type_): triangle index array of start function; see optri
        _mu (_type_): the measure that we want to achieve
        _sol (_type_): the exact solution values for the vertices, used to calculate inf-norm error
        infostring (string): string that will be displayed in the console
        max_iterations (int): maximum number of iterations

    Returns:
        _u (np.ndarray): updated function values
        _coords (np.ndarray): the coordinates of the vertices (identical to the input)
        _triags (np.ndarray): updated triangulation array
        _countADJ (np.ndarray): updated adjacency counter array
        _adj (np.ndarray): updated adjacency array
        _triagIDX (np.ndarray): updated triangle index array
        inf_norm (np.ndarray): array of inf-norm errors per iteration
        nbr_iterations (_type_): number of iterations needed
    """
    np.set_printoptions(precision=16)
    monge_ampere_precision = 1e-6
    inf_norm = []
    nbr_iterations = 0
    for repetitions in range(max_iterations):
        innerIndices = np.arange(_coords.shape[0])[~_bound]
        new_z = np.copy(_u)

        ###########################################################
        # adjust z values by Monge-Ampère-Measure calculation
        ###########################################################

        calculatedIndices = 0
        for index in innerIndices:
            if True:
                monge_ampere_measure = optri.calculateMongeAmpereMeasure(index, _adj,
                                                                         _triags, _countADJ, _coords, _u)
                if np.abs(monge_ampere_measure-_mu[index]) < monge_ampere_precision:
                    # print(f"index {index} skipped")
                    continue
            new_z[index] = ollikerPrussnerStep(index, _adj, _triags,
                                               _countADJ, _triagIDX, _coords, _u, _mu)
            calculatedIndices = calculatedIndices + 1
        if calculatedIndices == 0:
            # break iteration if Monge-Ampère-Measure is equal to the given measure mu (up to tollerance)
            # for all inner vertices
            opplot.bigprint("no more update done, iteration finished")
            nbr_iterations = repetitions
            break

        ###########################################################
        # globally overrite the z values
        ###########################################################
        _u = new_z

        ###########################################################
        # recalculate the triangulation with the new z values
        ###########################################################

        hullP = np.vstack((_coords.T, [_u])).T
        hull = optri.getLowerConvexHull(hullP, 2+repetitions*3, False)
        print("lower convex hull calculated")

        remainingIndices = np.setdiff1d(innerIndices, hull.vertices)
        print("remaining points: ", remainingIndices)

        points, triangles = optri.refineTriangulation(
            hull.simplices, hull.points, remainingIndices)
        _u = np.copy(points[:, 2])

        _triags = optri.sortTrianglesCCW(_coords, np.copy(triangles))
        _countADJ, _adj, _triagIDX = optri.findAdjacencies(
            np.copy(_coords), _triags)
        ###########################################################
        # calculate inf norm error and print it
        ###########################################################
        inf_norm.append(opplot.pNormError(_sol, _u, np.inf))
        opplot.bigprint(
            f"{infostring} it:{repetitions}\n\t{repetitions}-th inf norm error: {opplot.pNormError(_sol, _u, np.inf)}")

    return _u, _coords, _triags, _countADJ, _adj, _triagIDX, inf_norm, nbr_iterations


def load_data(pathname):
    """loads the data generated by the oliker_prussner_init.py
    """
    coordinates = np.loadtxt(pathname+"\\coordinates.dat", dtype=np.float64)
    boundaryMask = np.loadtxt(pathname+"\\boundaryMask.dat", dtype=np.bool_)
    muFromPoints = np.loadtxt(pathname+"\\muFromPoints.dat", dtype=np.float64)
    solution = np.loadtxt(pathname+"\\solution.dat", dtype=np.float64)
    u = np.loadtxt(pathname+"\\init.dat", dtype=np.float64)
    triangles = np.loadtxt(pathname+"\\elements.dat", dtype=np.int64)
    adjacencies = np.load(pathname+"\\adjacencies.npy")
    countADJ = np.loadtxt(pathname+"\\countADJ.dat", dtype=np.int64)
    triangleIdx = np.loadtxt(pathname+"\\elementIdx.dat", dtype=np.int64)
    solutionMu = np.loadtxt(pathname+"\\solutionMu.dat", dtype=np.float64)
    solutionTriags = np.loadtxt(
        pathname+"\\solutionTriags.dat", dtype=np.int64)
    return coordinates, boundaryMask, muFromPoints, solution, u, triangles, adjacencies, countADJ, triangleIdx, solutionMu, solutionTriags, pathname


def run(coordinates, boundaryMask, muFromPoints, _sol, u, triangles,
        adjacencies, countADJ, triangleIdx, solutionMu, solutionTriags, infostring, plot=False, measure_selection="solution"):
    """A wrapper function for managing and running the Oliker-Prussner-Method

    Args:
        coordinates (_type_): loaded from oliker_prussner_main.load_data
        boundaryMask (_type_): loaded from oliker_prussner_main.load_data
        muFromPoints (_type_): loaded from oliker_prussner_main.load_data
        _sol (_type_): loaded from oliker_prussner_main.load_data
        u (_type_): loaded from oliker_prussner_main.load_data
        triangles (_type_): loaded from oliker_prussner_main.load_data
        adjacencies (_type_): loaded from oliker_prussner_main.load_data
        countADJ (_type_): loaded from oliker_prussner_main.load_data
        triangleIdx (_type_): loaded from oliker_prussner_main.load_data
        solutionMu (_type_): loaded from oliker_prussner_main.load_data
        solutionTriags (_type_): loaded from oliker_prussner_main.load_data
        infostring (_type_): loaded from oliker_prussner_main.load_data
        plot (bool, optional): whether to plot the resulting function and inf-norms. Defaults to False.
        measure_selection (str, optional): whether to use the perfect Monge-Ampère-Measure ("solution")
            as referece for the method or the one calculated from the right hand side f("calculated"). Defaults to "solution".

    Raises:
        ValueError: if measure_selection is neither "solution" nor "calculated"
    """
    ##########################################
    # control area
    ##########################################
    max_iterations = 5000

    if measure_selection == "solution":
        _mu = solutionMu
    elif measure_selection == "calculated":
        _mu = muFromPoints
    else:
        raise ValueError("measure selection not valid")

    calculated_u, calculated_coords, calculated_triags, calculated_countADJ, calculated_adj, \
        calculated_triagIDX, inf_norm, nbr_iterations = oliker_prussner_sceme(
            coordinates, boundaryMask, adjacencies, triangles, countADJ, u, triangleIdx, _mu, _sol, infostring, max_iterations)

    ###########################################################
    # plotting the results
    ###########################################################
    inf_norm = np.array(inf_norm)
    np.savez(os.path.join(f"{infostring}", f"latest_run.npz"), u=calculated_u,
             coords=calculated_coords, triags=calculated_triags, countADJ=calculated_countADJ,
             adj=calculated_adj, triagIDX=calculated_triagIDX, inf_norm=inf_norm, nbr_iterations=nbr_iterations,
             solution=_sol, sol_mu=solutionMu, used_mu=_mu)

    if plot:
        plt.figure(100)
        opplot.show(calculated_triags, calculated_coords, calculated_u, figureNumber=100,
                    title="calculated solution")
        plt.figure(101)
        opplot.show(calculated_triags, calculated_coords, np.abs(calculated_u-_sol), figureNumber=101,
                    title="pointwise absolute distance to exact solution")
        plt.figure(102)
        plt.title("inf-norm-error per iterations")
        plt.plot(np.arange(1, inf_norm.size+1, 1),
                 inf_norm, ls="None", marker="o")
        plt.yscale("log")

    print(f"used parameters: {infostring}")
    print(f"final inf-norm error: {inf_norm[-1]}")
    print("final iteration: ", nbr_iterations)


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print(
            "usage: python oliker_prussner_main.py <path_to_folder> <\"solution\"|\"calculated\"> [<plot=False>]")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"path {path} does not exist")
        sys.exit(1)

    measure_selection = sys.argv[2]
    if measure_selection not in ["solution", "calculated"]:
        print(f"measure selection {sys.argv[2]} not valid")
        sys.exit(1)

    plot = False
    if len(sys.argv) == 4:
        if sys.argv[3] == "True" or sys.argv[3] == "true":
            plot = True

    opplot.bigprint("Oliker Prussner Method")

    run(*load_data(path), plot=plot,
        measure_selection=measure_selection)

    if plot:
        plt.show()
