import types
import _matrices
from _matrices import *




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
  


class MatrixType:
  def isA(cls, object):
    for tc in _TYPECODES:
      try:
        t = eval(cls.name()+_getTypeCodeName(tc))
        if type(object) is t:
          return True
      except:
        pass
    return False
  isA = classmethod(isA)
      
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
class SparseSymmetricExecuteMatrix(MatrixType):
  def name(): return "SparseSymmetricExecuteMatrix"
  name = staticmethod(name)
class SparseHermitianExecuteMatrix(MatrixType):
  def name(): return "SparseHermitianExecuteMatrix"
  name = staticmethod(name)
class SparseSymmetricBuildMatrix(MatrixType):
  def name(): return "SparseSymmetricBuildMatrix"
  name = staticmethod(name)
class SparseHermitianBuildMatrix(MatrixType):
  def name(): return "SparseHermitianBuildMatrix"
  name = staticmethod(name)




_TYPES = [
  Vector,
  DenseMatrix,
  SparseBuildMatrix,
  SparseExecuteMatrix,
  SparseSymmetricExecuteMatrix,
  SparseHermitianExecuteMatrix,
  SparseSymmetricBuildMatrix,
  SparseHermitianBuildMatrix 
  ]






# class getter ----------------------------------------------------------------
def makeRightType(module, name_trunk, typecode):
  return module.__dict__[name_trunk + _getTypeCodeName(typecode)]




def _getMatrixClass(rank, typecode, matrix_type):
  if rank == 1:
    typename = "Vector"
  else:
    if rank != 2:
      raise RuntimeError, "rank must be one or two"

    typename = matrix_type.name()
  return makeRightType(_matrices, typename, typecode)




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
      for i in data[0]:
        prev_biggest_type = getBiggestType(i, prev_biggest_type)
      return prev_biggest_type
    except:
      for i in data:
        if type(i) is types.ComplexType:
          prev_biggest_type = Complex
      return prev_biggest_type

  rank = getRank(data)
  if typecode is None:
    typecode = getBiggestType(data)

  mat_class = _getMatrixClass(rank, typecode, DenseMatrix)
  if rank == 2:
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
    return matrix_class(shape[0])
  else:
    return matrix_class(shape[0], shape[1])

def ones(shape, typecode, matrix_type = DenseMatrix):
  return _getFilledMatrix(shape, typecode, matrix_type, 1)

def identity(n, typecode, matrix_type = DenseMatrix):
  matrix_class = _getMatrixClass(2, typecode, matrix_type)
  return matrix_class.getIdentityMatrix(n)




# other functions -------------------------------------------------------------
def diagonal(mat, offset = 0):
  raise RuntimeError, "diagonal: not yet implemented"
  h,w = mat.shape
  if offset >= min(h,w):
    raise ValueError, "diagonal: invalid offset"
  if h>w:
    diag_length = 0
    pass
  else:
    pass


 

