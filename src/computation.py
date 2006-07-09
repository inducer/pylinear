"""
PyLinear's Python wrapper/functionality module for computational routines.
"""




import math, types
import pylinear
import pylinear.array as num
import pylinear._operation as _op




# computational routines ------------------------------------------------------
def solve_linear_system(mat, rhs):
    typecode = mat.typecode()
    h,w = mat.shape
    if mat.flavor is num.SparseExecuteMatrix and pylinear.has_umfpack():
        # use UMFPACK
        umf_operator = UMFPACKOperator.make(mat)

        temp = num.zeros((h,), typecode)
        if len(rhs.shape) == 1:
            umf_operator.apply(rhs, temp)
            return temp
        else:
            rhh, rhw = rhs.shape

            solution = num.zeros(rhs.shape, typecode)
            assert rhh == h
            for col in range(rhw):
                umf_operator.apply(rhs[:,col], temp)
                solution[:,col] = temp
            return solution
    else:
        # use lu
        l, u, permut, sign = lu(mat)

        temp = num.zeros((h,), typecode)
        if len(rhs.shape) == 1:
            for i in range(h):
                temp[i] = rhs[permut[i]]
            return u.solve_upper(l.solve_lower(temp))
        else:
            rhh, rhw = rhs.shape
        
            solution = num.zeros(rhs.shape, typecode)
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




def inverse(mat):
    w,h = mat.shape
    assert h == w
    return solve_linear_system(mat, num.identity(h, mat.typecode()))




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




def orthogonalize(vectors):
    # Gram-Schmidt FIXME: unstable
    done_vectors = []

    for v in vectors:
        my_v = v.copy()
        for done_v in done_vectors:
            my_v -= (v*done_v.H) * done_v
        v_norm = norm_2(my_v)
        if v_norm == 0:
            raise RuntimeError, "Orthogonalization failed"
        my_v /= v_norm
        done_vectors.append(my_v)
    return done_vectors




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
        right_hand_operator = None,
        spectral_shift = None,
        which_eigenvalues = LARGEST_MAGNITUDE,
        n_arnoldi_vectors = None,
        tolerance = 1e-12,
        max_iterations = None):

        if n_arnoldi_vectors is None:
            n_arnoldi_vectors = 2 * n_eigenvectors + 1

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
                               which_eigenvalues,
                               tolerance,
                               max_iterations)

        return zip(result.RitzValues, result.RitzVectors)




# matlab-workalikes -----------------------------------------------------------
def linspace(x, y, n = 100):
    if type(x) is types.IntType:
        x = float(x)
    if type(y) is types.IntType:
        y = float(y)
    h = (y-x) / n
    return [ x+h*i for i in range(n+1) ]




# other helpers ---------------------------------------------------------------
def make_permutation_matrix(permutation, typecode=num.Float):
    size = len(permutation)
    result = num.zeros((size,size), typecode)
    for index, value in zip(range(size), permutation):
        result[index,value] = 1
    return result



  
