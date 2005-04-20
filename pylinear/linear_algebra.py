import pylinear.array as num
import pylinear.operation as op



solve_linear_equations = op.solve_linear_system
inverse = op.inverse
determinant = op.determinant
singular_value_decomposition = op.svd
cholesky_decomposition = op.cholesky

eigenvalues = op.eigenvalues
eigenvectors = op.diagonalize
Heigenvalues = op.eigenvalues_hermitian
Heigenvectors = op.diagonalize_hermitian
