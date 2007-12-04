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
PyLinear's Python wrapper/functionality module for computational routines.
"""




import math, types
import pylinear
import pylinear.array as num
import pylinear._operation as _op
from pylinear._fft import *




# computational routines ------------------------------------------------------
def solve_linear_system(mat, rhs):
    dtype = mat.dtype
    h,w = mat.shape
    if mat.flavor is num.SparseExecuteMatrix and pylinear.has_umfpack():
        # use UMFPACK
        umf_operator = UMFPACKOperator.make(mat)

        temp = num.zeros((h,), dtype)
        if len(rhs.shape) == 1:
            umf_operator.apply(rhs, temp)
            return temp
        else:
            rhh, rhw = rhs.shape

            solution = num.zeros(rhs.shape, dtype)
            assert rhh == h
            for col in range(rhw):
                umf_operator.apply(rhs[:,col], temp)
                solution[:,col] = temp
            return solution
    else:
        # use lu
        l, u, permut, sign = lu(mat)

        temp = num.zeros((h,), dtype)
        if len(rhs.shape) == 1:
            for i in range(h):
                temp[i] = rhs[permut[i]]
            return u.solve_upper(l.solve_lower(temp))
        else:
            rhh, rhw = rhs.shape
        
            solution = num.zeros(rhs.shape, dtype)
            assert rhh == h
            for col in range(rhw):
                for i in range(h):
                    temp[i] = rhs[permut[i],col]
                    solution[:,col] = u.solve_upper(l.solve_lower(temp))
            return solution




def solve_linear_system_cg(matrix, vector):
    m_inv = CGOperator.make(MatrixOperator.make(matrix))
    return m_inv(vector)
    



cholesky = _op.cholesky
lu = _op.lu
if pylinear.has_lapack():
    svd = _op.singular_value_decomposition
    left_right_diagonalize = _op.eigenvectors

    def eigenvalues(mat):
        w, vl, vr = _op.eigenvectors(False, False, mat)
        return w

    def diagonalize(mat):
        w, vl, vr = _op.eigenvectors(False, True, mat)
        return vr, w

    def eigenvalues_hermitian(mat, upper = True):
        q, w = _op.Heigenvectors(False, upper, mat)
        return w

    def diagonalize_hermitian(mat, upper = True):
        return _op.Heigenvectors(True, upper, mat)

    def pseudo_inverse(mat):
        "Compute the Moore-Penrose Pseudo-Inverse of the argument."
        u, s_vec, vt = svd(mat)
        def inv_if_can(x):
            if x != 0:
                return 1./x
            else:
                return 0.
        s_vec_inv = num.array([inv_if_can(x) for x in s_vec])
        return vt.H * num.diagonal_matrix(s_vec_inv, shape=mat.shape[::-1]) * u.H





def inverse(mat):
    w,h = mat.shape
    assert h == w
    return solve_linear_system(mat, num.identity(h, mat.dtype))




def determinant(mat):
    h,w = mat.shape
    assert h == w
    if h == 2:
        return mat[0,0]*mat[1,1] - mat[1,0]*mat[0,1]
    else:
        try:
            l,u, permut, sign = _op.lu(mat)
            
            product = 1
            for i in range(h):
                product *= u[i,i]

            return product * sign
        except RuntimeError:
            # responds to the "is singular" exception
            return 0





def spectral_radius(mat):
    return max(eigenvalues(mat))




def spectral_condition_number(matrix, min_index = 0, threshold = None):
    u, s_vec, vt = svd(matrix)
    s_list = list(s_vec)
    s_list.sort(lambda a, b: cmp(abs(a), abs(b)))
    i = min_index
    if threshold is not None:
        while abs(s_list[i]) < threshold and i < len(s_list):
            i += 1
        if i == len(s_list):
            raise ValueError, "there is no eigenvalue above the threshold"
    return abs(s_list[-1])/abs(s_list[i])




def orthonormalize(vectors, discard_threshold=None):
    """Carry out a modified [1] Gram-Schmidt orthonormalization on
    vectors.

    If, during orthonormalization, the 2-norm of a vector drops 
    below C{discard_threshold}, then this vector is silently 
    discarded. If C{discard_threshold} is C{None}, then no vector
    will ever be dropped, and a zero 2-norm encountered during
    orthonormalization will throw an L{OrthonormalizationError}.

    [1] U{http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process}
    """

    done_vectors = []

    for v in vectors:
        my_v = v.copy()
        for done_v in done_vectors:
            my_v -= (my_v*done_v.H) * done_v
        v_norm = norm_2(my_v)

        if discard_threshold is None:
            if v_norm == 0:
                raise RuntimeError, "Orthogonalization failed"
        else:
            if v_norm < discard_threshold:
                continue

        my_v /= v_norm
        done_vectors.append(my_v)

    return done_vectors




def make_onb_with(vectors, dim=None, orth_threshold=1e-13):
    """Complete C{vectors} into an orthonormal basis.

    C{vectors} are verified to be orthogonal already. If empty,
    C{dim}, the dimension of the desired vector space, must be
    given.
    """
    from pytools import delta

    # first, find (and verify) dim
    for x in vectors:
        if dim is None:
            dim = len(x)
        else:
            if dim != len(x):
                raise ValueError, "not all vectors have same dimensionality"

    assert len(vectors) <= dim
    
    # next, assert given vectors are pairwise orthogonal
    for i, xi in enumerate(vectors):
        for j, yj in enumerate(vectors):
            assert abs(xi*yj - delta(i,j)) < orth_threshold

    vectors = vectors[:]

    for i in range(dim):
        vectors.append(num.unit_vector(dim, i))

    return orthonormalize(vectors, orth_threshold)




    
    






# matrix norms ----------------------------------------------------------------
def norm_spectral(mat):
    u, s_vec, vt = svd(mat)
    return max(num.absolute_value(s_vec))

def norm_frobenius_squared(mat):
    return mat.abs_square_sum()

def norm_frobenius(mat):
    return math.sqrt(mat.abs_square_sum())




# vector norms ----------------------------------------------------------------
def norm_1(vec):
    return num.absolute(vec).sum()

def norm_2_squared(vec):
    try:
        return vec.abs_square_sum()
    except AttributeError:
        return abs(vec)**2

def norm_2(vec):
    try:
        return math.sqrt(vec.abs_square_sum())
    except AttributeError:
        return math.sqrt(abs(vec)**2)

def norm_infinity(vec):
    # FIXME a tad slow
    return max(num.absolute(vec))




# arpack interface ------------------------------------------------------------
if pylinear.has_arpack():
    SMALLEST_MAGNITUDE = _op.SMALLEST_MAGNITUDE
    LARGEST_MAGNITUDE = _op.LARGEST_MAGNITUDE
    SMALLEST_REAL_PART = _op.SMALLEST_REAL_PART
    LARGEST_REAL_PART = _op.LARGEST_REAL_PART
    SMALLEST_IMAGINARY_PART = _op.SMALLEST_IMAGINARY_PART
    LARGEST_IMAGINARY_PART = _op.LARGEST_IMAGINARY_PART

    def operator_eigenvectors(
        operator,
        n_eigenvectors,
        right_hand_operator=None,
        spectral_shift=None,
        which=LARGEST_MAGNITUDE,
        n_arnoldi_vectors=None,
        tolerance=1e-12,
        max_iterations=None):

        if n_arnoldi_vectors is None:
            n_arnoldi_vectors = min(2 * n_eigenvectors + 1, operator.size1())

        mode = _op.REGULAR_NON_GENERALIZED
        if right_hand_operator is not None:
            mode = _op.REGULAR_GENERALIZED
        if spectral_shift is not None:
            mode = _op.SHIFT_AND_INVERT_GENERALIZED

        if max_iterations is None:
            max_iterations = 0

        result = _op.runArpack(operator, right_hand_operator,
                               mode, spectral_shift or 0,
                               n_eigenvectors,
                               n_arnoldi_vectors,
                               which,
                               tolerance,
                               max_iterations)

        return zip(result.RitzValues, result.RitzVectors)
