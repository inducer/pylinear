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
PyLinear's Python wrapper module for creating/manipulating Arrays.
(Array here means Matrix or Vector)
"""




from __future__ import division
from pylinear._array import *




# element type aliases --------------------------------------------------------
class Float64:
    pass
class Complex64:
    pass
Float = Float64
Complex = Complex64




# type code-related -----------------------------------------------------------
DTYPES = [
    Float64,
    Complex64
    ]




def _dtype_name(dtype):
    if dtype == Float64:
        return "Float64"
    elif dtype == Complex64:
        return "Complex64"
    else:
        raise RuntimeError, "Invalid dtype specified"

  


def _max_dtype(list):
    if Complex in list:
        return Complex
    else:
        return Float




class ParameterizedType(object):
    """
    A base class for "types" that depend on a dtype.

    This is a rather internal class.
    """

    def __init__(self, name, use_dict = None):
        if use_dict is None:
            use_dict = globals()
        self.Name = name

        type_dict = {}
        for tc in DTYPES:
            type_dict[tc] = use_dict[name + _dtype_name(tc)]
        self.TypeDict = type_dict

    def is_a(self, object):
        try:
            return isinstance(object, self(object.dtype))
        except NameError:
            return False

    def __str__(self):
        return self.Name

    def __call__(self, dtype):
        return self.TypeDict[dtype]

    def make(self, dtype, *args, **kwargs):
        return self.TypeDict[dtype](*args, **kwargs)

    def get_name(self):
        return self.Name
    name = property(get_name)






# flavor-related --------------------------------------------------------------
Vector = ParameterizedType("Vector")
DenseMatrix = ParameterizedType("Matrix")
SparseBuildMatrix = ParameterizedType("SparseBuildMatrix")
SparseExecuteMatrix = ParameterizedType("SparseExecuteMatrix")




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
        if matrix.dtype == Float:
            m_cast = asarray(matrix, Complex)
            return getattr(m_cast, "_nocast_" + operation)(other)
        else:
            print "CONFUSED"
            return NotImplemented

    try:
        other.flavor
    except AttributeError:
        return NotImplemented

    tcmax = _max_dtype([matrix.dtype, other.dtype])
    m_cast = asarray(matrix, tcmax)
    if other.flavor is Vector:
        o_cast = asarray(other, tcmax)
    else:
        o_cast = asarray(other, tcmax, matrix.flavor)
    return getattr(m_cast, "_nocast_" + operation)(o_cast)




def _vector_cast_and_retry(vector, operation, other):
    if _is_number(other):
        if vector.dtype == Float:
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

    tcmax = _max_dtype([vector.dtype, other.dtype])
    m_cast = asarray(vector, tcmax)
    o_cast = asarray(other, tcmax)
    return getattr(m_cast, "_nocast_" + operation)(o_cast)




def _matrix_power(x, n):
    if n < 0:
        x = 1 / x
        n *= -1

    # http://c2.com/cgi/wiki?IntegerPowerAlgorithm
    aux = identity(x.shape[0], x.dtype)
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




# equality testing ------------------------------------------------------------
def _equal(a, b):
    try:
        if a.shape != b.shape:
            return False
        diff = a - b
        for i in diff.indices():
            if diff[i] != 0:
                return False
        return True
    except AttributeError:
        return False

def _not_equal(a, b):
    return not _equals(a, b)





# array interface -------------------------------------------------------------
def _dtype_to_array_typestr(tc):
    import sys
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

    for dtype in DTYPES:
        dtype_array_typestr = _dtype_to_array_typestr(dtype)
        for f in FLAVORS:
            co = f(dtype)
            co.__add__ = co._ufunc_add
            co.__radd__ = co._ufunc_add
            co.__sub__ = co._ufunc_subtract
            co.__rsub__ = co._reverse_ufunc_subtract

            co.__eq__ = _equal
            co.__ne__ = _not_equal
            
            co.flavor = property(get_returner(f))
            co.dtype = property(get_returner(dtype))
            co.typecode = get_returner(dtype)
            
            co.__array_typestr__ = property(get_returner(dtype_array_typestr))


        DenseMatrix(dtype).__pow__ = _matrix_power
        DenseMatrix(dtype).__rdiv__ = _divide_by_matrix
        DenseMatrix(dtype).__rtruediv__ = _divide_by_matrix

        # stringification -----------------------------------------------------
        Vector(dtype).__str__ = _str_vector
        Vector(dtype).__repr__ = _repr_vector
        DenseMatrix(dtype).__str__ = _str_dense_matrix
        DenseMatrix(dtype).__repr__ = _repr_dense_matrix

        for f in FLAVORS:
            if f not in [Vector, DenseMatrix]:
                co = f(dtype)
                co.__str__ = _str_sparse_matrix
                co.__repr__ = _repr_sparse_matrix

        # cast_and_retry ------------------------------------------------------
        for mt in MATRIX_FLAVORS:
            mt(dtype)._cast_and_retry = _matrix_cast_and_retry
        for vt in VECTOR_FLAVORS:
            vt(dtype)._cast_and_retry = _vector_cast_and_retry




_add_array_behaviors()




# class getter ----------------------------------------------------------------
def _get_matrix_class(dim, dtype, flavor):
    if dim == 1:
        type_obj = Vector
    else:
        if dim != 2:
            raise RuntimeError, "dim must be one or two"

        type_obj = flavor or DenseMatrix
    return type_obj(dtype)




# construction functions ------------------------------------------------------
def array(data, dtype=None, flavor=None):
    """Create an Array from a (potentially nested) list of data values.

    Takes into account the given dtype and flavor. If None are specified,
    the minimum that will accomodate the given data are used.
    
    dtype can be one of Float,Complex, Float64, Complex64.
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

    if dtype is None:
        dtype = get_biggest_type(data)
    if flavor is None:
        flavor = DenseMatrix

    if dim == 2:
        mat_class = flavor(dtype)
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
        mat_class = Vector(dtype)
        h = len(data)
        result = mat_class(h)
        for i in range(h):
            result[i] = data[i]
        return result
    else:
        raise ValueError, "Invalid number of dimensions"




def arange(*args, **kwargs):
    """arange([start,] stop[, step,], dtype=None)
 
    For integer arguments, just like range() except it returns an array whose
    type can be specified by the keyword argument dtype.
 
    If dtype is not specified, the type of the result is deduced from the type
    of the arguments.
 
    For floating point arguments, the length of the result is ceil((stop -
    start)/step).  This rule may result in the last element of the result be
    greater than stop. 
    """
    # Yuck. Thanks NumPy developers, love y'all. ;)

    # argument processing -----------------------------------------------------
    start = None
    stop = None
    step = None
    dtype = None

    if args[-1] in DTYPES:
        dtype = args[-1]
        args = args[:-1]

    argc = len(args)
    if argc == 0:
        raise ValueError, "stop argument required"
    elif argc == 1:
        stop = args[0]
    elif argc == 2:
        start = args[0]
        stop = args[1]
    elif argc == 3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise ValueError, "too many arguments"

    admissible_names = ["start", "stop", "step", "dtype"]
    for k, v in kwargs.iteritems():
        if k in admissible_names:
            if locals()[k] is None:
                locals()[k] = v
            else:
                raise ValueError, "may not specify 'dtype' by position and keyword" % k
        else:
            raise ValueError, "unexpected keyword argument '%s'" % k

    if start is None:
        start = 0
    if step is None:
        step = 1

    # actual functionality ----------------------------------------------------
    import math
    length = int(math.ceil(float(stop-start)/step))
    return array([start + i*step for i in range(length)], dtype=dtype)




def linspace(start, stop, num=100, endpoint=True):
    """Return evenly spaced numbers.
 
    Return num evenly spaced samples from start to stop.  If
    endpoint is True, the last sample is stop. If retstep is
    True then return the step value used.
    """
    if endpoint:
        h = (stop-start) / (num-1)
    else:
        h = (stop-start) / num
        
    return [ start+h*i for i in range(num) ]




def sparse(mapping, shape=None, dtype=None, flavor=SparseBuildMatrix): 
    """Create a sparse Array from (two-level) nested
    mappings (e.g. dictionaries).

    Takes into account the given dtype and flavor. If None are specified,
    the minimum that will accomodate the given data are used.
    If shape is unspecified, the smallest size that can accomodate the data
    is used.

    See array() for valid dtype and flavors.
    """

    def get_biggest_type(mapping, prev_biggest_type = Float64):
        for row in mapping.values():
            for val in row.values():
                if isinstance(val, complex):
                    prev_biggest_type = Complex
        return prev_biggest_type

    if dtype is None:
        dtype = get_biggest_type(mapping)

    if shape is None:
        height = max(mapping.keys()) + 1
        width = 1
        for row in mapping.values():
            width = max(width, max(row.keys())+1)

        shape = height, width

    mat = flavor(dtype)(shape[0], shape[1])
    for i, row in mapping.iteritems():
        for j, val in row.iteritems():
            mat[i,j] = val
    return mat




def asarray(data, dtype=None, flavor=None):
    """Construct an array from data.
    
    Same as array(), except that a copy is made only when necessary.
    """

    try:
        given_flavor = data.flavor
        given_dtype = data.dtype
    except AttributeError:
        given_flavor = None
        given_dtype = None

    if flavor is None:
        flavor = given_flavor

    if given_dtype == dtype and given_flavor == flavor:
        return data

    if dtype is None and given_dtype is not None:
        dtype = given_dtype

    try:
        mat_class = _get_matrix_class(len(data.shape), dtype, flavor)
        return mat_class(data)
    except TypeError:
        return array(data, dtype, flavor)
  




def _get_filled_matrix(shape, dtype, matrix_type, fill_value):
    matrix_class = _get_matrix_class(len(shape), dtype, matrix_type)
    if len(shape) == 1:
        return matrix_class._get_filled_matrix(shape[0], fill_value)
    else:
        return matrix_class._get_filled_matrix(shape[0], shape[1], fill_value)

def empty(shape, dtype=Float, flavor=None):
    """Return an uninitialized array.
    
    This is often faster than zeros().
    """
    matrix_class = _get_matrix_class(len(shape), dtype, flavor)
    if len(shape) == 1:
        result = matrix_class(shape[0])
    else:
        result = matrix_class(shape[0], shape[1])
    return result

def empty_like(ary, shape=None, dtype=None, flavor=None):
    """Return an uninitialized array with shape, flavor and dtype just
    like `ary'.
    
    This is often faster than zeros_like().
    """
    result = empty(shape or ary.shape, dtype or ary.dtype, flavor or ary.flavor)
    return result

def zeros(shape, dtype=Float, flavor=None):
    """Return a zero-filled array."""
    result = empty(shape, dtype, flavor)
    result.clear()
    return result

def zeros_like(ary, shape=None, dtype=None, flavor=None):
    """Return a zero-filled array with shape, flavor and dtype just
    like `ary'.
    """
    result = zeros(shape or ary.shape, dtype or ary.dtype, flavor or ary.flavor)
    result.clear()
    return result

def ones(shape, dtype=Float, flavor=None):
    """Return a matrix filled with ones."""
    return _get_filled_matrix(shape, dtype, flavor, 1)

def eye(n, m=None, offset=0, dtype=Float, flavor=None):
    """Return a matrix `n' rows and `m' columns with the `offset'-th
    diagonal all ones, and everything else zeros.
    
    If `m' is None, it is assumed identical to `n'.
    """
    if m is None:
        m = n
    result = zeros((n,m), dtype, flavor)
    for i in range(max(0,-offset), min(m-offset,n)):
        result[i,i+offset] = 1
    return result

def tri(n, m=None, offset=0, dtype=Float, flavor=None):
    """Return a matrix `n' rows and `m' columns with the `offset'-th
    diagonal and the entires below it all ones, and everything else zeros.
    
    If `m' is None, it is assumed identical to `n'.
    """
    if m is None:
        m = n
    result = zeros((n,m), dtype, flavor)
    min_idx = max(0,-offset)
    for col in range(0,min_idx+offset):
        result[:,col] = 1
    for i in range(min_idx, min(m-offset,n)):
        result[i:,i+offset] = 1
    return result

def identity(n, dtype=Float, flavor=None):
    """Return an identity matrix.
    
    Deprecated in favor of the more powerful eye().
    """
    return eye(n, dtype=dtype, flavor=flavor)

def diagonal_matrix(vec_or_mat, shape=None, dtype=None, flavor=DenseMatrix):
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
        result = zeros(shape, dtype or vec.dtype,
                    flavor)
        for i in range(min((n,)+shape)):
            result[i,i] = vec[i]
        return result
    else:
        mat = vec_or_mat
        result = zeros(mat.shape, mat.dtype, mat.flavor)
        n = mat.shape[0]
        for i in range(n):
            result[i,i] = mat[i,i]
        return result

def hstack(tup, flavor=DenseMatrix):
    """Stack arrays in sequence horizontally (column wise)
     
    Description:
        Take a sequence of arrays and stack them horizontally
        to make a single array.  All arrays in the sequence
        must have the same shape along all but the second axis.
        hstack will rebuild arrays divided by hsplit.
    Arguments:
        tup -- sequence of arrays.  All arrays must have the same
               shape.
    """
    # don't check other array's height--the assignment will catch it if it's
    # wrong.
    h = tup[0].shape[0]
    w = sum([arr.shape[1] for arr in tup])

    result = zeros((h,w), _max_dtype(tup), flavor=flavor)

    index = 0
    for arr in tup:
        result[:,index:index+arr.shape[1]] = arr
        index += arr.shape[1]
    return result

def vstack(tup, flavor=DenseMatrix):
    """Stack arrays in sequence vertically (row wise)
 
    Description:
        Take a sequence of arrays and stack them veritcally
        to make a single array.  All arrays in the sequence
        must have the same shape along all but the first axis.
        vstack will rebuild arrays divided by vsplit.
    Arguments:
        tup -- sequence of arrays.  All arrays must have the same
               shape.
    """
    # don't check other array's width--the assignment will catch it if it's
    # wrong.
    w = tup[0].shape[1]
    h = sum([arr.shape[0] for arr in tup])

    result = zeros((h,w), _max_dtype(tup), flavor=flavor)

    index = 0
    for arr in tup:
        result[index:index+arr.shape[0]] = arr
        index += arr.shape[0]
    return result
    
def vsplit(ary, indices_or_sections):
    """Split ary into multiple rows of sub-arrays
 
    Description:
        Split a single array into multiple sub arrays.  The array is
        divided into groups of rows.  If indices_or_sections is
        an integer, ary is divided into that many equally sized sub arrays.
        If it is impossible to make the sub-arrays equally sized, the
        operation throws a ValueError exception. See array_split and
        split for other options on indices_or_sections.
    Arguments:
       ary -- N-D array.
          Array to be divided into sub-arrays.
       indices_or_sections -- integer or 1D array.
          If integer, defines the number of (close to) equal sized
          sub-arrays.  If it is a 1D array of sorted indices, it
          defines the indexes at which ary is divided.  Any empty
          list results in a single sub-array equal to the original
          array.
    Returns:
        sequence of sub-arrays.  The returned arrays have the same
        number of dimensions as the input array.
    """
    try:
        indices = indices_or_sections
        result = []
        last_end_index = 0
        for i, index in enumerate(indices):
            result.append(ary[last_end_index:index,:])
            last_end_index = index

        result.append(ary[last_end_index:])
        return result
    except TypeError:
        sections = indices_or_sections

        h,w = ary.shape

        if h % sections != 0:
            raise ValueError, "partitions are not of equal size"

        section_size = h/sections

        return vsplit(ary, range(section_size,h,section_size))

def hsplit(ary, indices_or_sections):
    """Split ary into multiple columns of sub-arrays
     
    Description:
        Split a single array into multiple sub arrays.  The array is
        divided into groups of columns.  If indices_or_sections is
        an integer, ary is divided into that many equally sized sub arrays.
        If it is impossible to make the sub-arrays equally sized, the
        operation throws a ValueError exception. See array_split and
        split for other options on indices_or_sections.
    Arguments:
       ary -- N-D array.
          Array to be divided into sub-arrays.
       indices_or_sections -- integer or 1D array.
          If integer, defines the number of (close to) equal sized
          sub-arrays.  If it is a 1D array of sorted indices, it
          defines the indexes at which ary is divided.  Any empty
          list results in a single sub-array equal to the original
          array.
    Returns:
        sequence of sub-arrays.  The returned arrays have the same
        number of dimensions as the input array.
    """
    try:
        indices = indices_or_sections
        result = []
        last_end_index = 0
        for i, index in enumerate(indices):
            result.append(ary[:,last_end_index:index])
            last_end_index = index

        result.append(ary[:,last_end_index:])
        return result
    except TypeError:
        sections = indices_or_sections

        h,w = ary.shape

        if w % sections != 0:
            raise ValueError, "partitions are not of equal size"

        section_size = w/sections

        return hsplit(ary, range(section_size,w,section_size))






# other functions -------------------------------------------------------------
def diagonal(mat, offset=0):
    """Return the (off-) diagonal of a matrix as a vector."""
    n,m = mat.shape

    min_idx = max(0, -offset)
    max_idx = min(n, m-offset)
    length = max_idx - min_idx

    if length < 0:
        raise ValueError, "diagonal: invalid offset"

    result = zeros((length,),  mat.dtype)
    for i in range(min_idx, max_idx):
        result[i-min_idx] = mat[i, i+offset]
    return result

def triu(mat, offset=0):
    """Return the upper right part of a matrix, up to the `offset'th super-diagonal.
    
    `offset' may be negative, indicating subdiagonals.
    """
    result = zeros(mat.shape, mat.dtype, mat.flavor)
    m, n = mat.shape 
    max_idx = min(m-offset,n)
    for i in range(max(0,-offset), max_idx):
        result[i,i+offset:] = mat[i,i+offset:]
    for col in range(max_idx, m):
        result[:,col] = 1
    return result

def tril(mat, offset=0):
    """Return the lower left part of a matrix, up to the `offset'th super-diagonal.
    
    `offset' may be negative, indicating subdiagonals.
    """
    result = zeros(mat.shape, mat.dtype, mat.flavor)
    m, n = mat.shape 
    min_idx = max(0,-offset)
    for col in range(0,min_idx+offset):
        result[:,col] = 1
    for i in range(min_idx, min(m-offset,n)):
        result[i:,i+offset] = mat[i:,i+offset]
    return result

def take(mat, indices, axis=0):
    if axis == 1:
        return array([mat[:,i] for i in indices], mat.dtype).T
    else:
        return array([mat[i] for i in indices], mat.dtype)
  
def matrixmultiply(mat1, mat2):
    """Multiply mat1 and mat2. For compatibility with NumPy."""
    return mat1 * mat2

def dot(ary1, ary2):
    """Multiply ary1 and ary2. For compatibility with NumPy."""
    return ary1 * ary2

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

_original_sum = sum
def sum(arr):
    """Return the sum of arr's entries."""
    try:
        return arr.sum()
    except AttributeError:
        return _original_sum(arr)


def product(arr, axis):
    """Return the product of arr's entries."""

    if axis is not None:
        raise ValueError, "product only supports axis=None."
    return arr._product_nonzeros()

def kroneckerproduct(a, b):
    """Return the Kronecker product of the two arguments.

    [[ a[ 0 ,0]*b, a[ 0 ,1]*b, ... , a[ 0 ,n-1]*b  ],
     [ ...                                   ...   ],
     [ a[m-1,0]*b, a[m-1,1]*b, ... , a[m-1,n-1]*b  ]]

    The result has the correct dtype for a product of the
    arguments and b's flavor.
    """
    ah, aw = a.shape
    bh, bw = b.shape
    tc = _max_dtype([a.dtype, b.dtype])
    result = zeros((ah*bh,aw*bw), tc, flavor=b.flavor)
    for i in range(ah):
        for j in range(aw):
            result[i*bh:(i+1)*bh,j*bw:(j+1)*bw] = a[i,j] * b
    return result





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
kron = _InfixOperator(kroneckerproduct)

def _solve_operator(mat, rhs):
    import pylinear.computation as comp
    return comp.solve_linear_system(mat, rhs)

solve = _InfixOperator(_solve_operator)

