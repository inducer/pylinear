import types
from pylinear._matrices import *




# element type aliases --------------------------------------------------------
Float = Float64
Complex = Complex64




# matrix types ----------------------------------------------------------------
_TYPECODES = [
    Float64,
    Complex64
    ]




def _getTypeCodeName(typecode):
    if typecode == Float64:
        return "Float64"
    elif typecode == Complex64:
        return "Complex64"
    else:
        raise RuntimeError, "Invalid typecode specified"
  


def _typeCodeMax(a, b):
    if Complex in [a, b]:
        return Complex
    else:
        return Float




class MatrixType:
    def isA(cls, object):
        try:
            return isinstance(object, cls.classObject(object.typecode()))
        except NameError:
            return False
    isA = classmethod(isA)

    def classObject(cls, typecode):
        return globals()[cls.name() + _getTypeCodeName(typecode)]
    classObject = classmethod(classObject)
      
class Vector(MatrixType):
    def name(): return "Vector"
    name = staticmethod(name)
class DenseMatrix(MatrixType):
    def name(): return "Matrix"
    name = staticmethod(name)
class SparseBuildMatrix(MatrixType):
    def name(): return "SparseBuildMatrix"
    name = staticmethod(name)
class SparseExecuteMatrix(MatrixType):
    def name(): return "SparseExecuteMatrix"
    name = staticmethod(name)




_MATRIX_TYPES = [
    DenseMatrix,
    SparseBuildMatrix,
    SparseExecuteMatrix,
    ]
_VECTOR_TYPES = [
    Vector,
    ]
_TYPES = _MATRIX_TYPES + _VECTOR_TYPES




def _is_number(value):
    try: 
        complex(value)
        return True
    except TypeError:
        return False




def _is_matrix(value):
    try: 
        _getMatrixType(value)
        return True
    except RuntimeError:
        return False




def _matrix_cast_and_retry(matrix, operation, other):
    if _is_number(other):
        if matrix.typecode() == Float:
            m_cast = asarray(matrix, Complex)
            return getattr(m_cast, "_nocast__%s__" % operation)(other)
        else:
            print "CONFUSED"
            return NotImplemented

    tcmax = _typeCodeMax(matrix.typecode(), other.typecode())
    m_cast = asarray(matrix, tcmax)
    if _getMatrixType(other) is Vector:
        o_cast = asarray(other, tcmax)
    else:
        mtype = _getMatrixType(matrix)
        o_cast = asarray(other, tcmax, mtype)
    return getattr(m_cast, "_nocast__%s__" % operation)(o_cast)




def _vector_cast_and_retry(vector, operation, other):
    if _is_number(other):
        if matrix.typecode() == Float:
            v_cast = asarray(vector, Complex)
            return getattr(v_cast, "_nocast__%s__" % operation)(other)
        else:
            print "CONFUSED"
            return NotImplemented

    if _getMatrixType(other) is not Vector:
        return NotImplemented

    tcmax = _typeCodeMax(vector.typecode(), other.typecode())
    m_cast = asarray(vector, tcmax)
    o_cast = asarray(other, tcmax)
    return getattr(m_cast, "_nocast__%s__" % operation)(o_cast)




for tc in _TYPECODES:
    for mt in _MATRIX_TYPES:
        mt.classObject(tc)._cast_and_retry = _matrix_cast_and_retry
    for vt in _VECTOR_TYPES:
        vt.classObject(tc)._cast_and_retry = _vector_cast_and_retry






# Tools -----------------------------------------------------------------------
def _maxTypeCode(list):
    if Complex in list:
        return Complex
    else:
        return Float




# class getter ----------------------------------------------------------------
def _getMatrixClass(rank, typecode, matrix_type):
    if rank == 1:
        type_obj = Vector
    else:
        if rank != 2:
            raise RuntimeError, "rank must be one or two"

        type_obj = matrix_type
    return type_obj.classObject(typecode)




def _getMatrixType(data):
    for t in _TYPES:
        if t.isA(data):
            return t
    raise RuntimeError, "Apparently not a matrix type"




# construction functions ------------------------------------------------------
def array(data, typecode = None):
    # slow, but that doesn't matter so much
    def getRank(data):
        try:
            return 1+getRank(data[0])
        except:
            return 0

    def getBiggestType(data, prev_biggest_type = Float64):
        try:
            data[0][0]
            for i in data:
                prev_biggest_type = getBiggestType(i, prev_biggest_type)
            return prev_biggest_type
        except TypeError, e:
            for i in data:
                if type(i) is types.ComplexType:
                    prev_biggest_type = Complex
            return prev_biggest_type

    rank = getRank(data)
    if typecode is None:
        typecode = getBiggestType(data)

    if rank == 2:
        mat_class = DenseMatrix.classObject(typecode)
        h = len(data)
        if h == 0:
            return mat_class(0, 0)
        w = len(data[0])
        result = mat_class(h, w)
        for i in range(h):
            for j in range(w):
                result[i,j] = data[i][j]
        return result
    elif rank == 1:
        mat_class = Vector.classObject(typecode)
        h = len(data)
        result = mat_class(h)
        for i in range(h):
            result[i] = data[i]
        return result



def asarray(data, typecode, matrix_type = None):
    given_matrix_type = _getMatrixType(data)
    if matrix_type is None:
        matrix_type = given_matrix_type
    if data.typecode() == typecode and given_matrix_type == matrix_type:
        return data
    mat_class = _getMatrixClass(len(data.shape), typecode, matrix_type)
    return mat_class(data)
  



def _getFilledMatrix(shape, typecode, matrix_type, fill_value):
    matrix_class = _getMatrixClass(len(shape), typecode, matrix_type)
    if len(shape) == 1:
        return matrix_class.getFilledMatrix(shape[0], fill_value)
    else:
        return matrix_class.getFilledMatrix(shape[0], shape[1], fill_value)

def zeros(shape, typecode, matrix_type = DenseMatrix):
    matrix_class = _getMatrixClass(len(shape), typecode, matrix_type)
    if len(shape) == 1:
        result = matrix_class(shape[0])
    else:
        result = matrix_class(shape[0], shape[1])
    result.clear()
    return result

def ones(shape, typecode, matrix_type = DenseMatrix):
    return _getFilledMatrix(shape, typecode, matrix_type, 1)

def identity(n, typecode, matrix_type = DenseMatrix):
    result = zeros((n,n), typecode, matrix_type)
    for i in range(n):
        result[i,i] = 1
    return result




# other functions -------------------------------------------------------------
def diagonal(mat, offset = 0):
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

 


def take(mat, indices, axis = 0):
    return array([mat[i] for i in indices], mat.typecode())
  
  


def matrixmultiply(mat1, mat2):
    return mat1 * mat2




def innerproduct(vec1, vec2):
    return vec1 * vec2




def outerproduct(vec1, vec2):
    try:
        return vec1._internal_outerproduct(vec2)
    except:
        if vec1.typecode() == vec2.typecode():
            raise

        mtc = _maxTypeCode([vec1.typecode(), vec2.typecode()])
        return outerproduct(asarray(vec1, mtc), asarray(vec2, mtc))





def transpose(mat):
    return mat.T




def hermite(mat):
    return mat.H




def trace(mat, k = 0):
    diag = diagonal(mat, k)
    return sum(diag)




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

def add(ma, mb): 
    try:
        return ma._ufunc_add(mb)
    except (AttributeError, TypeError):
        return mb._ufunc_add(ma)

def subtract(ma, mb): 
    try:
        return ma._ufunc_subtract(mb)
    except (AttributeError, TypeError):
        return mb._reverse_ufunc_subtract(ma)

def multiply(ma, mb): 
    try:
        return ma._ufunc_multiply(mb)
    except (AttributeError, TypeError):
        return mb._ufunc_multiply(ma)

def divide(ma, mb): 
    try:
        return ma._ufunc_divide(mb)
    except (AttributeError, TypeError):
        return mb._reverse_ufunc_divide(ma)

def divide_safe(ma, mb): 
    try:
        return ma._ufunc_divide_safe(mb)
    except (AttributeError, TypeError):
        return mb._reverse_ufunc_divide_safe(ma)

def power(ma, mb): 
    try:
        return ma._ufunc_power(mb)
    except (AttributeError, TypeError):
        return mb._reverse_ufunc_power(ma)

def maximum(ma, mb): 
    try:
        return ma._ufunc_maximum(mb)
    except (AttributeError, TypeError):
        return mb._ufunc_maximum(ma)

def minimum(ma, mb): 
    try:
        return ma._ufunc_minimum(mb)
    except (AttributeError, TypeError):
        return mb._ufunc_maximum(ma)

