"""
docstring
"""

import numpy as np
import numba as nb
from numba.experimental import jitclass
from .utils import interpolate, _boundary_condition, boundary_derivative


spec = [('N', nb.int64),
        ('numvars', nb.int64),
        ('bulk', nb.uint64[:]),
        ('edge', nb.uint64[:]),
        ('boundary_conditions', nb.float64[:, :]),
        ('state_vars', nb.float64[:]),
        ('rhs', nb.float64[:]),
        ('coeffs', nb.float64[:]),
        ('loc_indices_bulk', nb.int64[:, :, :]),
        ('loc_indices_edge', nb.int64[:, :, :]),
        ('jacobi', nb.float64[:, :, :]),
        ('sources', nb.float64[:])]



@jitclass(spec)
class DiffEqjit():
    """
    Jit-compiled class to store variables related
    to the solutions of the 1D diffusion equation.
    It is meant as an alternative to the C-style
    structs or similar data containers. We initialize
    the container to default values.
    Parameters:
    -----------

    N: int64
        Number of control volumes in the discretization scheme.

    numvars: int64
        Number of different state variables. For instance, if we
        have temperature T and concentration T, numvars = 2.

    loc_indices: ndarray, 2D
        Array of local indices to which a chosen control volume couples.
        Consider the following scheme for i-th control volume in the
        equation for a given state variable:
                           i-1      i      i+1
        |   o   |   o   |   o   |   o   |   o   |   o   |
        This indicates, that in our model, i-th control volume couples
        to its nearest rightmost and leftmost neighbour. In this case,
        the loc_indices array would be np.array([-1, 1], dtype=np.float64).
        In general, the shape of the loc_indices array should be
        (numvars, m), where m is the number of terms in the discretization
        scheme.


    Attributes:
    -----------

    state_vars: ndarray, 1D, float64
        Array of state variables, of length N * numvars,
        where N is the number of control volumes and numvars
        is the number of different state variables. For instance,
        if we used temperature T and concentration c as our
        state variables, we would have numvars = 2. Combined with,
        say, N=10, the state_vars array would store 20 elements.

    rhs: ndarray, 1D, float64
        Array of right-hand sides in the Newton iteration used to solve the
        nonlinear matrix system.

    loc_indices: ndarray, 3D, int64
        Array of shape (N, numvars, len(loc_indices)). For each control volume and
        each state variable, the array specifies couplings to other control volumes.

    loc_jac: ndarray, 3D, float64
        Array of the same shape as loc_indices. It specifies the Jacobian at each
        step of the iterative procedure.
    """

    def __init__(self, numvols: int, numvars: int, loc_indices: int):

        # ---------------------------------------
        #
        #  State variables
        #
        # ---------------------------------------
        self.N = numvols  # number of control volumes
        self.numvars = numvars  # number of different state variables

        # write for 1D structure here
        # define the edge indices
        self.edge = np.array([0, self.N - 1], dtype=np.uint64)
        # define the bulk indices
        self.bulk = np.arange(1, self.N - 1, dtype=np.uint64)
        # the boundary conditions: a 2D array
        self.boundary_conditions = np.zeros(
            (3, len(self.edge)), dtype=np.float64)

        # vector of state variables
        self.state_vars = np.zeros(numvols*numvars, dtype=np.float64)

        self.rhs = np.zeros_like(self.state_vars)  # vector of right-hand-sides
        self.coeffs = np.zeros_like(self.state_vars)  # transport coefficients
        self.sources = np.zeros_like(self.rhs)
        self._set_loc_indices(loc_indices)
        # ----------------------------------------
        #
        #  Matrix construction for PDE solvers.
        #
        # ----------------------------------------
        # self.loc_indices = np.array([loc_indices for i in range(
        #     N)], dtype=np.int64)  # indices in local jacobi terms
        # self.loc_jac = np.zeros(self.loc_indices.shape,
        #                         dtype=np.float64)  # local jacobi terms

    def _set_loc_indices(self, loc_indices):
        """
        An internal routine that sets an array of indices
        specifying with which neighbouring control volumes
        each control volume couples. We do so both for the
        bulk and edge control volumes which we prepare
        separately. At this point, code only handles the 1D
        case.
        """

        # number of elements in the bulk and the edge.
        bulk_size = len(self.bulk)
        edge_size = len(self.edge)

        nterms = len(loc_indices)

        # set local indices for bulk and edge control volumes
        # in the bulk in 1D, each control volume is specified by
        # n + 1 indices where n is the number of adjacent volumes
        # considered in the discretization scheme and the + 1
        # term describes the volume itself.
        self.loc_indices_bulk = np.zeros(
            (bulk_size, self.numvars, nterms), dtype=np.uint64)

        # in 1D structured mesh, there are only two boundary terms,
        # in the current scheme we only consider cases in which
        # the edge volume only couples to the nearest adjacent volume.
        self.loc_indices_edge = np.zeros(
            (edge_size, self.numvars, nterms-1), dtype=np.uint64)

        for i in range(bulk_size):
            self.loc_indices_bulk[i, :, :] = loc_indices

        for i in range(edge_size):
            self.loc_indices_edge[i, :, :] = np.array([(-1)**i])


@nb.njit(fastmath=True, nogil=True, cache=True, boundscheck=True)
def update_rhs(cls, delta_state: float, delta_x: float, delta_t: float):
    """
    A function to update the right-hand side of the iterative procedure

    """

    # initial step: sources and the old solution
    cls.rhs = -cls.sources - (cls.state_vars / delta_t)

    # update the old solution to get the proposition
    # for the new state/
    cls.state_vars += delta_state

    # add the new state to the iterative scheme
    cls.rhs += cls.state_vars / delta_t

    # the actual physical part begins here
    # iterate over bulk and edge terms
    it_be = [cls.bulk, cls.edge]  # iterate over bulk/edge terms
    it_be_idx = [cls.loc_indices_bulk, cls.loc_indices_edge]


    # ---------------------------------------------------------------
    # FILL THE GENERAL TERMS IN THE RHS VECTOR
    # ---------------------------------------------------------------
    for _it_idx, grid in enumerate(it_be):  # iterate over bulk/edge
        for _i, i in enumerate(grid):  # iterate over control volumes in bulk/edge

            for j in range(cls.numvars):  # iterate over state variables

                _idx = i + j * cls.N  # temporary index

                # iterate over neighbouring volumes to build the rhs
                for loc_ind in it_be_idx[_it_idx][_i, j, :]:

                    # interpolate the transport coefficients
                    _coeff = interpolate(
                        cls.coeffs[_idx], cls.coeffs[_idx + loc_ind])

                    # the central volume
                    cls.rhs[_idx] += cls.state_vars[_idx] * \
                        _coeff / delta_x**2
                    # contributions from the neighbours
                    cls.rhs[_idx] += cls.state_vars[_idx + loc_ind] * \
                        _coeff / delta_x**2

    # ---------------------------------------------------------------
    # SPECIAL CONTRIBUTIONS TO THE RHS vector on the edge
    # ---------------------------------------------------------------
    for i, edge_idx in enumerate(cls.edge):  # iterate over the edge volumes

        for var_idx in range(cls.numvars):

            tmp_idx = edge_idx + var_idx * cls.N

            bcarr = cls.boundary_conditions[:, i]
            neighbour = cls.loc_indices_edge[i, var_idx, 0]  # in 1D: \pm 1

            _rhs = _boundary_condition(cls.state_vars, cls.coeffs, bcarr, tmp_idx,
                                       neighbour, delta_x)

            cls.rhs[tmp_idx] = _rhs


def build_jac(jac, ):
    """
    A function to update the jacobi array
    """

    pass


if __name__ == '__main__':

    clsdiff = DiffEqjit(1000, 1, np.array([-1, 1], dtype=np.int64))
    clsdiff.state_vars = np.ones_like(clsdiff.state_vars)
    # print(clsdiff.loc_indices_edge)
    # print(clsdiff.bulk)
    update_rhs(clsdiff, 0.1*np.ones_like(clsdiff.state_vars), 0.1, 0.1)
