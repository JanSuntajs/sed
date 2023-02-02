"""
In this subpackage, we develop tools for numerically treating
the diffusion equation. For pedagogical purposes, we focus on
the 1D case with a regular structured mesh, but if time permits,
we also extend the analysis to 2D and the related unstructured
meshes.

We make use of the FDM and FEM schemes (finite difference methods
and finite volume methods) to discretize the grid and we implement
different methods to forward propagate the equation in time. We
ultimately implement the implicit Crank-Nicolson method as the
method of choice, but explicit and implicit Euler scheme are also
included.

Modules:

discretize.py: implements different discretization schemes

tevol.py: implements time evolution schemes

model.py: implements the physical model (partial differential equation)
          to be solved.

TO DO: implement some 3rd party solver and solve these equations in
its context; this is ultimately the preferred way of doing things -
why would we reinvent the wheel?
"""
