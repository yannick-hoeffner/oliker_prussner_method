######################################################################################################
#
#   Example usage of the Oliker-Prussner method for solving the Monge-Ampère-Equation.
#
#   by Yannick Höffner, Friedrich-Schiller-University Jena,
#   Bachelor Thesis WS 2024/25
#
#   based on the original paper from V.I.Oliker and L.D.Prussner:
#   "On the Numerical Solution of the Equation [...] and Its Discretizations, I"
#   Numerische Mathematik 54.3 (1989): 271-294
#
#
######################################################################################################
import oliker_prussner_method.plotting as opplot
import oliker_prussner_method.core as op_core
import oliker_prussner_method.initialisation as op_coreinit
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


########################################################################################################
#   CREATE A GRID
########################################################################################################


def create_square_points(n: int, beta: np.float64) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """creates the grid points for the square domain,
    where the boundary points are sorted to the beginning of the list

    Args:
        n (int): number of grid points per dimension
        beta (np.float64): scaling factor to generate a graduated grid

    Returns:
        coords (NDArray[float64]): list of x,y coordinates of the grid points
        bound (NDArray[bool]): mask for coords with 1 if the desired vertex is a boundary point
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

    # if you want to transform the coordinates, uncomment the following lines
    # print("transforming coordinates to [0,pi]x[0,pi]")
    # coordinates = coordinates + np.array([1, 1])
    # coordinates = coordinates * np.pi

    # sort boundary points to the beginning of the grid point list
    coordinates = np.append(
        coordinates[boundaryMask], coordinates[~boundaryMask], axis=0)
    boundaryMask = np.append(
        boundaryMask[boundaryMask], boundaryMask[~boundaryMask], axis=0)

    return coordinates, boundaryMask


def create_circular_points(ppc: int, nbrc: int) -> npt.NDArray[np.float64]:
    """generates a mesh of the unit circle consisting of
    multiple circles with a given number of points per circle,
    where the boundary points are sorted to the beginning of the list

    automatically adds a point at the origin
    Args:
        ppc (int): points per circle
        nbrc (int): number of circles

    Returns:
        coords (ndarray): list of x,y coordinates of the grid points
        bound (ndarray): mask for coords with 1 if the desired vertex is a boundary point
    """
    circle = np.linspace(0, 2*np.pi, ppc, endpoint=False, dtype=np.float64)
    radii = np.linspace(1, 0, nbrc, endpoint=False, dtype=np.float64)

    x = np.outer(radii, np.cos(circle)).flatten()
    y = np.outer(radii, np.sin(circle)).flatten()
    x = np.append(x, 0)
    y = np.append(y, 0)

    coordinates = np.stack((x, y)).T
    boundaryMask = np.append(np.ones(ppc, dtype=bool),
                             np.zeros(x.shape[0]-ppc, dtype=bool))

    # sort boundary points to the beginning of the grid point list
    coordinates = np.append(
        coordinates[boundaryMask], coordinates[~boundaryMask], axis=0)
    boundaryMask = np.append(
        boundaryMask[boundaryMask], boundaryMask[~boundaryMask], axis=0)

    return coordinates, boundaryMask


########################################################################################################
#   INFORMATION GIVEN THROUGH THE MONGE-AMPÈRE-EQUATION
########################################################################################################


def f(val: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """function for the right hand side of
    the Monge-Ampère-Equation

    Args:
        val (ndarray): array [[x1,y1],[x2,y2],...] with data from the domain

    Returns:
        ndarray: f(x,y) for every (x,y) pair as a list
    """
    # f(x,y) = 4 for all (x,y) in the domain
    # this is the right hand side of the Monge-Ampère-Equation with the
    # exact solution u(x,y) = x^2 + y^2
    return np.repeat(4, np.size(val, 0))


def calculate_boundary_condition(coordinates: npt.NDArray[np.float64], boundary_mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """calculates the boundary condition for the given grid points

    Args:
        coordinates (ndarray): list of x,y coordinates of the grid points
        boundary_mask (ndarray): mask for coords; valued 1 if the desired vertex is a boundary point

    Returns:
        boundary_condition (ndarray): boundary function values for the given grid points
    """
    boundary_condition = np.zeros(coordinates.shape[0], dtype=np.float64)
    boundary_condition[boundary_mask] = (coordinates[
        boundary_mask, 0] ** 2 + coordinates[boundary_mask, 1]**2)

    return boundary_condition


########################################################################################################
#   (OPTIONAL) TESTING AGAINST THE REAL SOLUTION
########################################################################################################

def exact_solution(coordinates: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # exact solution u(x,y) = x^2 + y^2
    return coordinates[:, 0]**2 + coordinates[:, 1]**2

########################################################################################################
#   DEMONSTRATION OF THE BASIC WORKFLOW
########################################################################################################


if __name__ == '__main__':
    # Step 1: Create the grid points/boundary mask
    coordinates, boundary_mask = create_square_points(9, 1.0)

    # Step 2: Create boundary condition
    boundary_condition = calculate_boundary_condition(
        coordinates, boundary_mask)

    # Step 3: Create the initial triangulation and function values
    init_triangulation, init_u_values = op_coreinit.create_init_triangulation(
        coordinates, boundary_mask, boundary_condition)
    plt.figure(10)
    plt.title("initial triangulation")
    plt.triplot(coordinates[:, 0], coordinates[:, 1], init_triangulation,
                marker="o", color="green", linewidth=2)
    opplot.show(init_triangulation, coordinates,
                init_u_values, 11, "initial function")

    # Step 4: create the Monge-Ampère-Measure, which the OP-method
    # should approximate (the measure is computed with the right hand side function f
    # of the Monge-Ampère-Equation)
    # Alternatively you can provide the measure directly as a list of values for each vertex
    measure = op_coreinit.compute_MA_measure(coordinates, boundary_mask, f)
    opplot.show(init_triangulation, coordinates,
                measure, 12, r"measure $\mu$")

    # Step 5: execute the OP-method
    z_vals, final_triags, nbr_iter = op_core.oliker_prussner_sceme(
        coordinates, boundary_mask, init_triangulation, init_u_values, measure, "test", 5000)
    print(f"Number of iterations needed: {nbr_iter}")

    opplot.show(final_triags, coordinates,
                z_vals, 13, "calculated function")

    # Step 6: (optional) calculate the infinity norm of the difference between the
    # calculated solution and the exact solution
    inf_norm = np.max(np.abs(z_vals - exact_solution(coordinates)))
    print(
        f"Infinity norm of the difference between calculated and exact solution: {inf_norm}")

    plt.show()
