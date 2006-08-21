#
#  Copyright (c) 2004-2006
#  Andreas Kloeckner
#
#  Permission to use, copy, modify, distribute and sell this software
#  and its documentation for any purpose is hereby granted without fee,
#  provided that the above copyright notice appear in all copies and
#  that both that copyright notice and this permission notice appear
#  in supporting documentation.  The authors make no representations
#  about the suitability of this software for any purpose.
#  It is provided "as is" without express or implied warranty.
#




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
