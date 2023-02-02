import numba as nb
import numpy as np

# intepolation
@nb.njit("float64(float64, float64)", fastmath=True,
         nogil=True, error_model='numpy')
def interpolate(val1, val2):
    """
    Interpolation scheme for the transport coefficients
    used in the finite volume method. We used
    the geometric averaging here so as to avoid
    situations in which a vanishing value of the
    transport coefficient in one control volume would
    still yield a nonvanishing transport coefficient
    after averaging. We have already considered the
    fact we are using a uniform mesh in 1D so this
    function is strictly tailored for this case.

    The interpolation sceheme is as follows:

    coeff = 2 * (val1 * val2) / (val1 + val2)

    Parameters:
    -----------

    val1, val2: float64
        Values to be interpolated

    Returns:
    --------

    coeff: float64
        The interpolated value.
    """
    if val1 == val2 == 0:

        return 0.

    else:
        coeff = 2 * val1 * val2 / (val1 + val2)

        return coeff



@nb.njit("float64(float64[:], float64[:], float64[:], int64, uint64, float64)",
         fastmath=True, nogil=True)
def _boundary_condition(state_vars, coeffs, bcarr,
                        edge, neighbour, delta_x):
    """
    An internal routine to properly treat the boundary conditions.
    Currently, Dirichlet and von Neumann bc are considered, might
    add Robin bc in the future.

    Parameters:
    -----------

    state_vars: ndarray, 1D, float64
        Array of state variables.

    coeffs: ndarray, 1D, float64
        Array of transport coefficients.

    bcarr: ndarray, 1D, float64
        Array specifying boundary conditions.
        Entries should be like this:
        bcarr[0]: 0. or 1., for Dirichlet or von Neumann
        boundary conditions, respectively.
        bcarr[1]: boundary value of the state variable
        bcarr[2]: boundary flux.

    edge: uint64
        Index of the edge control volume.

    neighbour: int64
        Difference between the index of the edge control
        volume and the index of the neighbouring control volume.

    delta_x: float64
        Discretization step in the discretization scheme.

    Returns:
    --------

    retval: float64
        Update of the right-hand side vector in the iteration scheme.

    """

    coeff = (3 * coeffs[edge] - coeffs[edge + neighbour]) * 0.5

    # ----------------------------------
    # DIRICHLET
    # ----------------------------------
    if bcarr[0] == 0.:  

        vals = (9 * state_vars[edge] - state_vars[edge +
                neighbour]) / (3. * delta_x ** 2)
        # add to rhs
        rhs = -(8./(3.*delta_x**2)) * coeff * bcarr[1]

        retval = vals * coeff + rhs
    # ----------------------------------
    # Von Neumann
    # ----------------------------------
    if bcarr[0] == 1.:  # von Neumann boundary condition

        retval = coeff * bcarr[2] / delta_x

    return retval


def boundary_derivative():
    """
    
    docstring
    """