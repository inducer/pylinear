"""
PyLinear's compability wrapper with NumPy's LinearAlgebra.

Do not use this for code written for PyLinear, use pylinear.computation
instead.
"""




import pylinear.array as num
import pylinear.computation as comp



solve_linear_equations = comp.solve_linear_system
inverse = comp.inverse
determinant = comp.determinant
singular_value_decomposition = comp.svd
cholesky_decomposition = comp.cholesky

eigenvalues = comp.eigenvalues
eigenvectors = comp.diagonalize
Heigenvalues = comp.eigenvalues_hermitian
Heigenvectors = comp.diagonalize_hermitian
