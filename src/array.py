"""
PyLinear's Python wrapper module for creating/manipulating Arrays.
(Array here means Matrix or Vector)
"""




import sys
from pylinear._array import *




def version():
    """Return a 3-tuple with the PyLinear version."""
    return (0,92,0)




# element type aliases --------------------------------------------------------
class Float64:
    pass
class Complex64:
    pass
Float = Float64
Complex = Complex64




# type code-related -----------------------------------------------------------
TYPECODES = [
    Float64,
    Complex64
    ]




def _typecode_name(typecode):
    if typecode == Float64:
        return "Float64"
    elif typecode == Complex64:
        return "Complex64"
    else:
        raise RuntimeError, "Invalid typecode specified"

  


def _max_typecode(list):
    if Complex in list:
        return Complex
    else:
        return Float




class TypecodeParameterizedType(object):
    """
    A base class for "types" that depend on a typecode.

    This is a rather internal class.
    """

    def __init__(self, name, use_dict = None):
        if use_dict is None:
            use_dict = globals()
        self.Name = name

        type_dict = {}
        for tc in TYPECODES:
            type_dict[tc] = use_dict[name + _typecode_name(tc)]
        self.TypeDict = type_dict

    def is_a(self, object):
        try:
            return isinstance(object, self(object.typecode()))
        except NameError:
            return False

    def __call__(self, typecode):
        return self.TypeDict[typecode]

    def make(self, typecode, *args, **kwargs):
        return self.TypeDict[typecode](*args, **kwargs)

    def get_name(self):
        return self.Name
    name = property(get_name)






# flavor-related --------------------------------------------------------------
Vector = TypecodeParameterizedType("Vector")
DenseMatrix = TypecodeParameterizedType("Matrix")
SparseBuildMatrix = TypecodeParameterizedType("SparseBuildMatrix")
SparseExecuteMatrix = TypecodeParameterizedType("SparseExecuteMatrix")




# additional array functionality ----------------------------------------------
MATRIX_FLAVORS = [
    DenseMatrix,
    SparseBuildMatrix,
    SparseExecuteMatrix,
    ]
VECTOR_FLAVORS = [
    Vector,
    ]
FLAVORS = MATRIX_FLAVORS + VECTOR_FLAVORS




def _is_number(value):
    try: 
        complex(value)
        return True
    except AttributeError:
        return False
    except TypeError:
        return False




def _is_matrix(value):
    try: 
        return value.flavor in FLAVORS
    except NameError:
        return False




def _matrix_cast_and_retry(matrix, operation, other):
    if _is_number(other):
        if matrix.typecode() == Float:
            m_cast = asarray(matrix, Complex)
            return getattr(m_cast, "_nocast_" + operation)(other)
        else:
            print "CONFUSED"
            return NotImplemented

    try:
        other.flavor
    except AttributeError:
        return NotImplemented

    tcmax = _max_typecode([matrix.typecode(), other.typecode()])
    m_cast = asarray(matrix, tcmax)
    if other.flavor is Vector:
        o_cast = asarray(other, tcmax)
    else:
        o_cast = asarray(other, tcmax, matrix.flavor)
    return getattr(m_cast, "_nocast_" + operation)(o_cast)




def _vector_cast_and_retry(vector, operation, other):
    if _is_number(other):
        if vector.typecode() == Float:
            v_cast = asarray(vector, Complex)
            return getattr(v_cast, "_nocast_" + operation)(other)
        else:
            print "CONFUSED"
            return NotImplemented

    try:
        if other.flavor is not Vector:
            return NotImplemented
    except AttributeError:
        return NotImplemented

    tcmax = _max_typecode([vector.typecode(), other.typecode()])
    m_cast = asarray(vector, tcmax)
    o_cast = asarray(other, tcmax)
    return getattr(m_cast, "_nocast_" + operation)(o_cast)




def _matrix_power(x, n):
    if n < 0:
        x = 1 / x
        n *= -1

    # http://c2.com/cgi/wiki?IntegerPowerAlgorithm
    aux = identity(x.shape[0], x.typecode())
    while n > 0:
        if n & 1: 
            aux *= x
            if n == 1: 
                return aux
        x = x * x
        n //= 2
    return aux




def _divide_by_matrix(mat, x):
    import pylinear.linear_algebra as la
    if not _is_number(x):
        return NotImplemented
    return la.inverse(mat) * x




# stringification -------------------------------------------------------------
def _wrap_vector(strs, max_length=80, indent=8*" ", first_indent=0):
    result = ""
    line_length = first_indent
    for i, s in enumerate(strs):
        item_length = len(s) + 2
        if line_length + item_length > max_length:
            line_length = len(indent)
            result += "\n%s" % indent
        line_length += item_length
        result += s
        if i != len(strs) - 1:
            result += ", "
    return result




def _stringify_vector(array, num_stringifier):
    strs = [num_stringifier(entry) for entry in array]
    return "array([%s])" % _wrap_vector(strs, indent=7*" ", first_indent=7)




def _stringify_dense_matrix(array, num_stringifier):
    h,w = array.shape
    strs = [[num_stringifier(array[i,j]) for i in range(h)] for j in range(w)]
    col_widths = [max([len(s) for s in col]) for col in strs]
    result = ""
    for i, v in enumerate(array):
        row = [strs[j][i].rjust(col_widths[j])
               for j in range(w)]
        result += "[%s]" % _wrap_vector(row, indent=12*" ", first_indent=8)
        if i != h - 1:
            result += ",\n" + 7 * " "
    return "array([%s])" % result

def _stringify_sparse_matrix(array, num_stringifier):
    strs = []
    last_row = -1
    for i, j in array.indices():
        if i != last_row:
            current_row = []
            strs.append((i, current_row))
            last_row = i

        current_row.append("%d: %s" % (j, num_stringifier(array[i,j])))

    result = ""
    for row_idx, (i,row) in enumerate(strs):
        indent = 10+len(str(row_idx))
        result += "%d: {%s}" % (i, _wrap_vector(row, indent=(indent + 4)*" ", first_indent=indent))
        if row_idx != len(strs) - 1:
            result += ",\n" + 8 * " "
    return "sparse({%s},\n%sshape=%s, flavor=%s)" % (
        result, 7*" ",repr(array.shape), array.flavor.name)

def _str_vector(array): return _stringify_vector(array, str)
def _str_dense_matrix(array): return _stringify_dense_matrix(array, str)
def _str_sparse_matrix(array): return _stringify_sparse_matrix(array, str)

def _repr_vector(array): return _stringify_vector(array, repr)
def _repr_dense_matrix(array): return _stringify_dense_matrix(array, repr)
def _repr_sparse_matrix(array): return _stringify_sparse_matrix(array, repr)




# array interface -------------------------------------------------------------
def _typecode_to_array_typestr(tc):
    if sys.byteorder == "big":
        indicator = ">"
    elif sys.byteorder == "little":
        indicator = "<"
    else:
        raise RuntimeError, "Invalid byte order detected."

    if tc is Float64:
        return indicator + "f8"
    elif tc is Complex64:
        return indicator + "c16"
    else:
        raise RuntimeError, "Invalid type code received."




# python-implemented methods --------------------------------------------------
def _add_array_behaviors():
    def get_returner(value):
        # This routine is necessary since we don't want the lambda in
        # the top-level scope, whose variables change.
        return lambda self: value

    for tc in TYPECODES:
        tc_array_typestr = _typecode_to_array_typestr(tc)
        for f in FLAVORS:
            co = f(tc)
            co.__add__ = co._ufunc_add
            co.__radd__ = co._ufunc_add
            co.__sub__ = co._ufunc_subtract
            co.__rsub__ = co._reverse_ufunc_subtract
            
            co.flavor = property(get_returner(f))
            co.typecode = get_returner(tc)
            
            co.__array_typestr__ = property(get_returner(tc_array_typestr))


        DenseMatrix(tc).__pow__ = _matrix_power
        DenseMatrix(tc).__rdiv__ = _divide_by_matrix
        DenseMatrix(tc).__rtruediv__ = _divide_by_matrix

        # stringification -----------------------------------------------------
        Vector(tc).__str__ = _str_vector
        Vector(tc).__repr__ = _repr_vector
        DenseMatrix(tc).__str__ = _str_dense_matrix
        DenseMatrix(tc).__repr__ = _repr_dense_matrix

        for f in FLAVORS:
            if f not in [Vector, DenseMatrix]:
                co = f(tc)
                co.__str__ = _str_sparse_matrix
                co.__repr__ = _repr_sparse_matrix

        # cast_and_retry ------------------------------------------------------
        for mt in MATRIX_FLAVORS:
            mt(tc)._cast_and_retry = _matrix_cast_and_retry
        for vt in VECTOR_FLAVORS:
            vt(tc)._cast_and_retry = _vector_cast_and_retry




_add_array_behaviors()




# class getter ----------------------------------------------------------------
def _get_matrix_class(dim, typecode, flavor):
    if dim == 1:
        type_obj = Vector
    else:
        if dim != 2:
            raise RuntimeError, "dim must be one or two"

        type_obj = flavor or DenseMatrix
    return type_obj(typecode)




# construction functions ------------------------------------------------------
def array(data, typecode=None, flavor=None):
    """Create an Array from a (potentially nested) list of data values.

    Takes into account the given typecode and flavor. If None are specified,
    the minimum that will accomodate the given data are used.
    
    typecode can be one of Float,Complex, Float64, Complex64.
    flavor can be one of Vector, DenseMatrix, SparseBuildMatrix, SparseExecuteMatrix.
    """
    # slow, but that doesn't matter so much
    def get_dim(data):
        try:
            return 1+get_dim(data[0])
        except:
            return 0

    def get_biggest_type(data, prev_biggest_type = Float64):
        try:
            data[0][0]
            for i in data:
                prev_biggest_type = get_biggest_type(i, prev_biggest_type)
            return prev_biggest_type
        except TypeError:
            for i in data:
                if isinstance(i, complex):
                    prev_biggest_type = Complex
            return prev_biggest_type

    dim = get_dim(data)

    if typecode is None:
        typecode = get_biggest_type(data)
    if flavor is None:
        flavor = DenseMatrix

    if dim == 2:
        mat_class = flavor(typecode)
        h = len(data)
        if h == 0:
            return mat_class(0, 0)
        w = len(data[0])
        result = mat_class(h, w)
        for i in range(h):
            for j in range(w):
                result[i,j] = data[i][j]
        return result
    elif dim == 1:
        mat_class = Vector(typecode)
        h = len(data)
        result = mat_class(h)
        for i in range(h):
            result[i] = data[i]
        return result
    else:
        raise ValueError, "Invalid number of dimensions"




def sparse(mapping, shape=None, typecode=None, flavor=SparseBuildMatrix):
    """Create a sparse Array from (two-level) nested mappings (e.g. dictionaries).

    Takes into account the given typecode and flavor. If None are specified,
    the minimum that will accomodate the given data are used.
    If shape is unspecified, the smallest size that can accomodate the data
    is used.

    See array() for valid typecodes and flavors.
    """
    def get_biggest_type(mapping, prev_biggest_type = Float64):
        for row in mapping.values():
            for val in row.values():
                if isinstance(val, complex):
                    prev_biggest_type = Complex
        return prev_biggest_type

    if typecode is None:
        typecode = get_biggest_type(mapping)

    if shape is None:
        height = max(mapping.keys()) + 1
        width = 1
        for row in mapping.values():
            width = max(width, max(row.keys())+1)

        shape = height, width

    mat = flavor(typecode)(shape[0], shape[1])
    for i, row in mapping.iteritems():
        for j, val in row.iteritems():
            mat[i,j] = val
    return mat




def asarray(data, typecode=None, flavor=None):
    """Construct an array from data.
    
    Same as array(), except that a copy is made only when necessary.
    """
    try:
        given_flavor = data.flavor
        given_tc = data.typecode()
    except NameError:
        given_flavor = None
        given_tc = None

    if flavor is None:
        flavor = given_flavor

    if given_tc == typecode and given_flavor == flavor:
        return data

    if typecode is None and given_tc is not None:
        typecode = given_tc

    try:
        mat_class = _get_matrix_class(len(data.shape), typecode, flavor)
        return mat_class(data)
    except TypeError:
        return array(data, typecode, flavor)
  




def _get_filled_matrix(shape, typecode, matrix_type, fill_value):
    matrix_class = _get_matrix_class(len(shape), typecode, matrix_type)
    if len(shape) == 1:
        return matrix_class._get_filled_matrix(shape[0], fill_value)
    else:
        return matrix_class._get_filled_matrix(shape[0], shape[1], fill_value)

def zeros(shape, typecode=Float, flavor=None):
    """Return a zero-filled array."""
    matrix_class = _get_matrix_class(len(shape), typecode, flavor)
    if len(shape) == 1:
        result = matrix_class(shape[0])
    else:
        result = matrix_class(shape[0], shape[1])
    result.clear()
    return result

def ones(shape, typecode=Float, flavor=None):
    """Return a matrix filled with ones."""
    return _get_filled_matrix(shape, typecode, flavor, 1)

def identity(n, typecode=Float, flavor=None):
    """Return an identity matrix."""
    result = zeros((n,n), typecode, flavor)
    for i in range(n):
        result[i,i] = 1
    return result

def diagonal_matrix(vec_or_mat, shape=None, typecode=None, flavor=DenseMatrix):
    """Return a given Array as a diagonal matrix.
    
    If vec_or_mat is a vector, return a diagonal matrix of the same size
    with the vector on the diagonal.
    
    If vec_or_mat is a matrix, return only its diagonal, but still in matrix
    shape."""
    if len(vec_or_mat.shape) == 1:
        vec = vec_or_mat
        n = vec.shape[0]
        if shape is None:
            shape = (n,n)
        result = zeros(shape, typecode or vec.typecode(),
                    flavor)
        for i in range(min((n,)+shape)):
            result[i,i] = vec[i]
        return result
    else:
        mat = vec_or_mat
        result = zeros(mat.shape, mat.typecode(), mat.flavor)
        n = mat.shape[0]
        for i in range(n):
            result[i,i] = mat[i,i]
        return result




# other functions -------------------------------------------------------------
def diagonal(mat, offset=0):
    """Return the (off-) diagonal of a matrix as a vector."""
    h,w = mat.shape

    result = []
    if offset >= 0:
        # upwards offset, i.e. to the right
        post_end_row = min(offset+h, w)
        length = post_end_row - offset
        result = zeros((length,),  mat.typecode())
        if length < 0:
            raise ValueError, "diagonal: invalid offset"
        
        for i in range(length):
            result[i] = mat[i, i+offset]
        return result
    else:
        # downwards offset
        offset = - offset
        post_end_col = min(offset+w, h)
        length = post_end_col - offset
        result = zeros((length,),  mat.typecode())
        if length < 0:
            raise ValueError, "diagonal: invalid offset"
        
        for i in range(length):
            result[i] = mat[i+offset, i]
        return result

def lower_left(mat, include_diagonal=False):
    """Return the lower left half of a matrix."""
    result = zeros(mat.shape, mat.typecode(), mat.flavor)
    if include_diagonal:
        for i,j in mat.indices():
            if i >= j:
                result.set_element_past_end(i, j, mat[i,j])
    else:
        for i,j in mat.indices():
            if i > j:
                result.set_element_past_end(i, j, mat[i,j])
    return result

def upper_right(mat, include_diagonal=False):
    """Return the upper right half of a matrix."""
    result = zeros(mat.shape, mat.typecode(), mat.flavor)
    if include_diagonal:
        for i,j in mat.indices():
            if i <= j:
                result.set_element_past_end(i, j, mat[i,j])
    else:
        for i,j in mat.indices():
            if i < j:
                result.set_element_past_end(i, j, mat[i,j])
    return result

def take(mat, indices, axis=0):
    if axis == 1:
        return array([mat[:,i] for i in indices], mat.typecode()).T
    else:
        return array([mat[i] for i in indices], mat.typecode())
  
def matrixmultiply(mat1, mat2):
    """Multiply mat1 and mat2. For compatibility with NumPy."""
    return mat1 * mat2

def innerproduct(vec1, vec2):
    """Multiply vec1 and vec2. For compatibility with NumPy."""
    return vec1 * vec2

def outerproduct(vec1, vec2):
    """Return the (matrix) outer product of vec1 and vec2."""
    return vec1._outerproduct(vec2)

def crossproduct(vec1, vec2):
    """Return the cross product of vec1 and vec2."""
    (v1len,) = vec1.shape
    (v2len,) = vec2.shape
    if v1len == 3 and v2len == 3:
        return array([
        vec1[1]*vec2[2]-vec1[2]*vec2[1],
        vec1[2]*vec2[0]-vec1[0]*vec2[2],
        vec1[0]*vec2[1]-vec1[1]*vec2[0]])
    elif v1len == 2 and v2len == 2:
        return vec1[0]*vec2[1]-vec1[1]*vec2[0]
    else:
        raise ValueError, "cross product requires two vectors of dimension 2 or 3"

def transpose(mat):
    """Return the transpose of mat. For compatibility with NumPy."""
    return mat.T

def hermite(mat):
    """Return the complex-conjugate transpose of mat. For compatibility with NumPy."""
    return mat.H

def trace(mat, offset=0):
    """Return the trace of a matrix."""
    diag = diagonal(mat, offset)
    return sum(diag)

def sum(arr):
    """Return the sum of arr's entries."""
    return arr.sum()

def product(arr, axis):
    """Return the product of arr's entries."""

    if axis is not None:
        raise ValueError, "product only supports axis=None."
    return arr._product_nonzeros()



# ufuncs ----------------------------------------------------------------------
def conjugate(m): 
    return m._ufunc_conjugate()
def cos(m): 
    """Return the elementwise cosine of the argument Array."""
    return m._ufunc_cos()
def cosh(m): 
    """Return the elementwise hyperbolic cosine of the argument Array."""
    return m._ufunc_cosh()
def exp(m): 
    """Return the elementwise base-e exponential of the argument Array."""
    return m._ufunc_exp()
def log(m): 
    """Return the elementwise base-e logarithm of the argument Array."""
    return m._ufunc_log()
def log10(m): 
    """Return the elementwise base-10 logarithm of the argument Array."""
    return m._ufunc_log10()
def log2(m): 
    """Return the elementwise base-2 of the argument Array."""
    import math
    return m._ufunc_log10() / math.log10(2)
def sin(m): 
    """Return the elementwise sine of the argument Array."""
    return m._ufunc_sin()
def sinh(m): 
    """Return the elementwise hyperbolic sine of the argument Array."""
    return m._ufunc_sinh()
def sqrt(m): 
    """Return the elementwise square root of the argument Array."""
    return m._ufunc_sqrt()
def tan(m): 
    """Return the elementwise tangent of the argument Array."""
    return m._ufunc_tan()
def tanh(m): 
    """Return the elementwise hyperbolic tangent of the argument Array."""
    return m._ufunc_tanh()
def floor(m): 
    """Return the elementwise floor of the argument Array."""
    return m._ufunc_floor()
def ceil(m): 
    """Return the elementwise ceiling of the argument Array."""
    return m._ufunc_ceil()
def arg(m): 
    """Return the elementwise complex argument of the argument Array."""
    return m._ufunc_arg()
angle = arg
def absolute(m): 
    """Return the elementwise absolute value of the argument Array."""
    return m._ufunc_absolute()

class _BinaryUfunc:
    def __call__(self, op1, op2):
        try:
            result = self.execute(op1, op2)
            if result is NotImplemented:
                return self.executeInReverse(op1, op2)
            else:
                return result
        except AttributeError:
            return self.executeInReverse(op1, op2)

    def executeInReverse(self, op1, op2):
        # default to commutative operations
        return self.execute(op2, op1)

class _BinaryAdd(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_add(op2)
class _BinarySubtract(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_subtract(op2)
    def executeInReverse(self, op1, op2): return op2._reverse_ufunc_subtract(op1)
class _BinaryMultiply(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_multiply(op2)
class _BinaryDivide(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_divide(op2)
    def executeInReverse(self, op1, op2): return op2._reverse_ufunc_divide(op1)
class _BinaryPower(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_power(op2)
    def executeInReverse(self, op1, op2): return op2._reverse_ufunc_power(op1)
class _BinaryMaximum(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_maximum(op2)
class _BinaryMinimum(_BinaryUfunc):
    def execute(self, op1, op2): return op1._ufunc_minimum(op2)

add = _BinaryAdd()
subtract = _BinarySubtract()
multiply = _BinaryMultiply()
divide = _BinaryDivide()
power = _BinaryPower()
maximum = _BinaryMaximum()
minimum = _BinaryMinimum()




# fake infix operators --------------------------------------------------------
class _InfixOperator:
    """Pseudo-infix operators that allow syntax of the kind `op1 <<operator>> op2'.
    
    Following a recipe from
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/384122
    """
    def __init__(self, function):
        self.function = function
    def __rlshift__(self, other):
        return _InfixOperator(lambda x: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def call(self, a, b):
        return self.function(a, b)

outer = _InfixOperator(outerproduct)
cross = _InfixOperator(crossproduct)

def _solve_operator(mat, rhs):
    import pylinear.computation as comp
    return comp.solve_linear_system(mat, rhs)

solve = _InfixOperator(_solve_operator)

