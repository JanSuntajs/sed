"""
This module implements a class for investigating the
diffusion in 1D.


TO DO: extend to 2D and allow for arbitrary meshes


"""
import numpy as np


class diffusion1D(object):
    """
    Creates a class to model the
    solutions of a general unsteady 1D diffusion
    equation of the form

    (d \phi)/(dt) = D (d^2 \phi) / (d \phi^2) + S_\phi

    Where D is a positive quantity (diffusion
    coefficient, which can, in general, be position,
    time and \phi dependent) and S_\phi models
    the source terms. 

    Attributes:
    -----------  

    x0: float,
        A float specifiying the initial value along the x-axis.

    L: float, 
        A float the domain length in 1D.

    N: int,
        An integer specifying the discretization of the
        domain (the number of pieces in which the domain
        is discretized).

    t: float, optional
        Initial time, set to zero by default.


    Methods:
    --------


    """

    def __init__(self, x0, L, N, phi0, t0=0,):
        """
        Initialize.

        Parameters
        ----------

        x0: float,
            A float specifiying the initial value along the x-axis.

        L: float,
            A float specifiying the domain length in 1D.

        N: int,
            An integer specifiying the discretization of the domain
            (the number of pieces in which the domain is discretized.)

        """
        super(diffusion1D, self).__init__()

        # grid details
        self.x0 = x0
        self.L = L
        self.N = N

        self.grid = np.linspace(self.x0, self.x0 + self.L, self.N)

        # initial profile
        self.phi = phi0

        # time evolution params
        self.t = t0

    # routines for setting and checking L, N, t

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):

        if L <= 0:
            raise ValueError("L must be greater than zero!")

        self._L = L

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        if type(N) is not int:
            raise TypeError("N must be an integer!")
        if N <= 0:
            raise ValueError("N must be greater than zero!")

        self._N = N

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):

        if t <= 0:
            raise ValueError("L must be greater than zero!")

        self._t = t

    @property
    def grid(self)

        return _grid
    
    @grid.setter
    def grid(self, grid):

        self._grid = np.linspace(self.x0, self.x0 + self.L, self.N)
