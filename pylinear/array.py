from pylinear._array import *




def version():
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

    if other.flavor is not Vector:
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




def _stringify_array(array):
    # slow? does it matter?

    def wrap_vector(strs, max_length=80, indent=8*" ", first_indent=0):
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

    if array.flavor is Vector:
        strs = [str(entry) for entry in array]
        return "array([%s])" % wrap_vector(strs, indent=7*" ", first_indent=7)
    elif array.flavor is DenseMatrix:
        h,w = array.shape
        strs = [[str(array[i,j]) for i in range(h)] for j in range(w)]
        col_widths = [max([len(s) for s in col]) for col in strs]
        result = ""
        for i, v in enumerate(array):
            row = [strs[j][i].rjust(col_widths[j])
                   for j in range(w)]
            result += "[%s]" % wrap_vector(row, indent=12*" ", first_indent=8)
            if i != h - 1:
                result += ",\n" + 7 * " "
        return "array([%s])" % result
    else: # sparse matrices
        strs = []
        last_row = -1
        for i, j in array.indices():
            if i != last_row:
                current_row = []
                strs.append((i, current_row))
                last_row = i

            current_row.append("%d: %s" % (j, str(array[i,j])))

        result = ""
        for row_idx, (i,row) in enumerate(strs):
            indent = 10+len(str(row_idx))
            result += "%d: {%s}" % (i, wrap_vector(row, indent=(indent + 4)*" ", first_indent=indent))
            if row_idx != len(strs) - 1:
                result += ",\n" + 8 * " "
        return "sparse({%s},\n%sshape=%s, flavor=%s)" % (
            result, 7*" ",repr(array.shape), array.flavor.name)




def _add_array_behaviors():
    def get_returner(value):
        # This routine is necessary since we don't want the lambda in
        # the top-level scope, whose variables change.
        return lambda self: value

    for tc in TYPECODES:
        for f in FLAVORS:
            co = f(tc)
            co.__add__ = co._ufunc_add
            co.__radd__ = co._ufunc_add
            co.__sub__ = co._ufunc_subtract
            co.__rsub__ = co._reverse_ufunc_subtract
            
            co.flavor = property(get_returner(f))
            co.typecode = get_returner(tc)
            
            co.__str__ = _stringify_array
            co.__repr__ = _stringify_array

        DenseMatrix(tc).__pow__ = _matrix_power
        DenseMatrix(tc).__rdiv__ = _divide_by_matrix
        DenseMatrix(tc).__rtruediv__ = _divide_by_matrix
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

def zeros(shape, typecode, flavor=None):
    matrix_class = _get_matrix_class(len(shape), typecode, flavor)
    if len(shape) == 1:
        result = matrix_class(shape[0])
    else:
        result = matrix_class(shape[0], shape[1])
    result.clear()
    return result

def ones(shape, typecode, flavor=None):
    return _get_filled_matrix(shape, typecode, flavor, 1)

def identity(n, typecode, flavor=None):
    result = zeros((n,n), typecode, flavor)
    for i in range(n):
        result[i,i] = 1
    return result

def diagonal_matrix(vec_or_mat, typecode=None, flavor=DenseMatrix):
    if len(vec_or_mat.shape) == 1:
        vec = vec_or_mat
        n = vec.shape[0]
        result = zeros((n, n), typecode or vec.typecode(),
                    flavor)
        for i in range(n):
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
    return mat1 * mat2

def innerproduct(vec1, vec2):
    return vec1 * vec2

def outerproduct(vec1, vec2):
    return vec1._outerproduct(vec2)

def transpose(mat):
    return mat.T

def hermite(mat):
    return mat.H

def trace(mat, offset=0):
    diag = diagonal(mat, offset)
    return sum(diag)

def sum(arr):
    return arr.sum()




# ufuncs ----------------------------------------------------------------------
def conjugate(m): return m._ufunc_conjugate()
def cos(m): return m._ufunc_cos()
def cosh(m): return m._ufunc_cosh()
def exp(m): return m._ufunc_exp()
def log(m): return m._ufunc_log()
def log10(m): return m._ufunc_log10()
def sin(m): return m._ufunc_sin()
def sinh(m): return m._ufunc_sinh()
def sqrt(m): return m._ufunc_sqrt()
def tan(m): return m._ufunc_tan()
def tanh(m): return m._ufunc_tanh()
def floor(m): return m._ufunc_floor()
def ceil(m): return m._ufunc_ceil()
def arg(m): return m._ufunc_arg()
def absolute(m): return m._ufunc_absolute()

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
    """Following a recipe from

    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/384122

    This allows, for example, vec1 <<outer>> vec2
    """
    def __init__(self, function):
        self.function = function
    def __rlshift__(self, other):
        return _InfixOperator(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)

outer = _InfixOperator(outerproduct)

def _solve_operator(mat, rhs):
    import pylinear.operation as op
    return op.solve_linear_system(mat, rhs)

solve = _InfixOperator(_solve_operator)

