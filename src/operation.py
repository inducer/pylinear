import math, types
import pylinear.array as num
import pylinear._operation as _op




# operator parameterized types ------------------------------------------------
Operator = num.TypecodeParameterizedType(
  "MatrixOperator", _op.__dict__)
IdentityOperator = num.TypecodeParameterizedType(
  "IdentityMatrixOperator", _op.__dict__)
ScalarMultiplicationOperator = num.TypecodeParameterizedType(
  "ScalarMultiplicationMatrixOperator", _op.__dict__)

class _MatrixOperatorTypecodeFlavorParameterizedType:
    def is_a(self, instance):
        # FIXME
        raise NotImplementedError

    def __call__(self, typecode, flavor):
        # FIXME
        raise NotImplementedError

    def make(self, matrix):
        return _op.makeMatrixOperator(matrix)
MatrixOperator = _MatrixOperatorTypecodeFlavorParameterizedType()

class _CGTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.typecode(), w)
        if matrix_op.typecode() is not precon_op.typecode():
            raise TypeError, "matrix_op and precon_op must have matching typecodes"
        return self.TypeDict[matrix_op.typecode()](matrix_op, precon_op, max_it, tolerance)
    
CGOperator = _CGTypecodeParameterizedType("CGMatrixOperator", _op.__dict__)

class _BiCGSTABTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.typecode(), w)
        if matrix_op.typecode() is not precon_op.typecode():
            raise TypeError, "matrix_op and precon_op must have matching typecodes"
        return self.TypeDict[matrix_op.typecode()](matrix_op, precon_op, max_it, tolerance)
    
BiCGSTABOperator = _BiCGSTABTypecodeParameterizedType(
    "BiCGSTABMatrixOperator", _op.__dict__)

if _op.has_umfpack():
    class _UMFPACKTypecodeParameterizedType(num.TypecodeParameterizedType):
        def make(self, matrix):
            matrix.complete_index1_data()
            return self.TypeDict[matrix.typecode()](matrix)

    UMFPACKOperator = _UMFPACKTypecodeParameterizedType("UMFPACKMatrixOperator", 
                                                        _op.__dict__)

class _LUInverseOperator:
    def __init__(self, l, u, perm):
        assert l.shape[0] == l.shape[1]
        assert u.shape[0] == u.shape[1]
        assert l.shape[0] == u.shape[0]

        self.L = l
        self.U = u
        self.Permutation = perm

    def size1(self):
        return self.L.shape[0]
    
    def size2(self):
        return self.L.shape[1]

    def apply(self, before, after):
        temp = num.zeros((len(before),), before.typecode())
        for i in range(len(before)):
            temp[i] = before[self.Permutation[i]]
        after[:] = self.U.solve_upper(self.L.solve_lower(temp))

class _LUInverseOperatorFloat64(_LUInverseOperator, _op.MatrixOperatorFloat64):
    def __init__(self, l, u, perm):
        _LUInverseOperator.__init__(self, l, u, perm)
        _op.MatrixOperatorFloat64.__init__(self)

class _LUInverseOperatorComplex64(_LUInverseOperator, _op.MatrixOperatorComplex64):
    def __init__(self, l, u, perm):
        _LUInverseOperator.__init__(self, l, u, perm)
        _op.MatrixOperatorComplex64.__init__(self)

class _LUInverseTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, *args):
        if len(args) == 3:
            l, u, perm = args
        elif len(args) == 1:
            l, u, perm, sign = lu(args[0])
        else:
            raise TypeError, "Invalid number of arguments"

        return self.TypeDict[l.typecode()](l, u, perm)

LUInverseOperator = _LUInverseTypecodeParameterizedType("_LUInverseOperator", 
                                                        globals())

class _SSORPreconditioner:
    def __init__(self, mat, omega=1):
        # mat needs to be symmetric
        assert mat.shape[0] == mat.shape[1]

        l = num.lower_left(mat)
        d = num.diagonal_matrix(mat)

        self.L = d + omega*l
        self.U = self.L.H
        self.DVector = num.diagonal(mat)
        self.Omega = omega

    def size1(self):
        return self.L.shape[0]
    
    def size2(self):
        return self.L.shape[1]

    def apply(self, before, after):
        after[:] = self.Omega * (2-self.Omega) * \
                   self.U.solve_upper(num.multiply(self.DVector, 
                                                  self.L.solve_lower(before)))

class _SSORPreconditionerFloat64(_SSORPreconditioner, 
                                 _op.MatrixOperatorFloat64):
    def __init__(self, *args, **kwargs):
        _SSORPreconditioner.__init__(self, *args, **kwargs)
        _op.MatrixOperatorFloat64.__init__(self)

class _SSORPreconditionerComplex64(_SSORPreconditioner, 
                                   _op.MatrixOperatorComplex64):
    def __init__(self, *args, **kwargs):
        _SSORPreconditioner.__init__(self, *args, **kwargs)
        _op.MatrixOperatorComplex64.__init__(self)

class _SSORPreconditionerTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, mat, *args, **kwargs):
        return num.TypecodeParameterizedType.make(
            self, mat.typecode(), mat, *args, **kwargs)

SSORPreconditioner = _SSORPreconditionerTypecodeParameterizedType(
    "_SSORPreconditioner", globals())


# operator operators ----------------------------------------------------------
_SumOfOperators = num.TypecodeParameterizedType(
  "SumOfMatrixOperators", _op.__dict__)
_ScalarMultiplicationOperator = num.TypecodeParameterizedType(
  "ScalarMultiplicationMatrixOperator", _op.__dict__)
_CompositeOfOperators = num.TypecodeParameterizedType(
  "CompositeMatrixOperator", _op.__dict__)




def _neg_operator(op):
    return _compose_operators(
        _ScalarMultiplicationOperator(op.typecode())(-1, op.shape[0]),
        op)

def _add_operators(op1, op2):
    return _SumOfOperators(op1.typecode())(op1, op2)

def _subtract_operators(op1, op2):
    return _add_operators(op1, _neg_operator(op2))

def _compose_operators(outer, inner):
    return _CompositeOfOperators(outer.typecode())(outer, inner)

def _multiply_operators(op1, op2):
    if num._is_number(op2):
        return _compose_operators(
            op1,
            _ScalarMultiplicationOperator(op1.typecode())(op2, op1.shape[0]))
    else:
        return _compose_operators(op1, op2)

def _reverse_multiply_operators(op1, op2):
    # i.e. op2 * op1
    assert num._is_number(op2)
    return _compose_operators(
        _ScalarMultiplicationOperator(op1.typecode())(op2, op1.shape[0]),
        op1)

def _call_operator(op1, op2):
    try:
        temp = num.zeros((op1.shape[0],), op2.typecode())
        op1.apply(op2, temp)
        return temp
    except TypeError:
        # attempt applying a real operator to a complex problem
        temp_r = num.zeros((op1.shape[0],), num.Float)
        temp_i = num.zeros((op1.shape[0],), num.Float)
        op1.apply(op2.real, temp_r)
        op1.apply(op2.imaginary, temp_i)
        return temp_r + 1j*temp_i




def _add_operator_behaviors():
    def get_returner(value):
        # This routine is necessary since we don't want the lambda in
        # the top-level scope, whose variables change.
        return lambda self: value

    for tc in num.TYPECODES:
        Operator(tc).__neg__ = _neg_operator
        Operator(tc).__add__ = _add_operators
        Operator(tc).__sub__ = _subtract_operators
        Operator(tc).__mul__ = _multiply_operators
        Operator(tc).__rmul__ = _reverse_multiply_operators
        Operator(tc).__call__ = _call_operator
        Operator(tc).typecode = get_returner(tc)




_add_operator_behaviors()





# library support queries -----------------------------------------------------
has_blas = _op.has_blas
has_lapack = _op.has_lapack
has_arpack = _op.has_arpack
has_umfpack = _op.has_umfpack




# computational routines ------------------------------------------------------
def solve_linear_system(mat, rhs):
    typecode = mat.typecode()
    h,w = mat.shape
    if mat.flavor is num.SparseExecuteMatrix and _op.has_umfpack():
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
if _op.has_lapack():
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
    return num.absolute_value(vec).sum()

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
    return max(num.absolute_value(vec))




# arpack interface ------------------------------------------------------------
if _op.has_arpack():
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



  
