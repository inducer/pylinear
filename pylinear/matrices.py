import types
import matrices_internal
from matrices_internal import *




# element type aliases --------------------------------------------------------
Float = Float64
Complex = Complex64




# matrix types ----------------------------------------------------------------
class DenseMatrix:
  def name(): return "DenseMatrix"
  name = staticmethod(name)
class SparseBuildMatrix:
  def name(): return "SparseBuildMatrix"
  name = staticmethod(name)
class SparseExecuteMatrix:
  def name(): return "SparseExecuteMatrix"
  name = staticmethod(name)
class SparseSymmetricExecuteMatrix:
  def name(): return "SparseSymmetricExecuteMatrix"
  name = staticmethod(name)
class SparseHermitianExecuteMatrix:
  def name(): return "SparseHermitianExecuteMatrix"
  name = staticmethod(name)
class SparseSymmetricBuildMatrix:
  def name(): return "SparseSymmetricBuildMatrix"
  name = staticmethod(name)
class SparseHermitianBuildMatrix:
  def name(): return "SparseHermitianBuildMatrix"
  name = staticmethod(name)




# class getter ----------------------------------------------------------------
def makeRightType(module, name_trunk, typecode):
  name = name_trunk
  if typecode == matrices_internal.Float64:
    name += "Float64"
  elif typecode == matrices_internal.Complex64:
    name += "Complex64"
  else:
    raise RuntimeError, "Invalid element type requested"

  return module.__dict__[name]




def _getMatrixClass(rank, typecode, matrix_type):
  if rank == 1:
    typename = "Vector"
  else:
    if rank != 2:
      print rank
      raise RuntimeError, "rank must be one or two"

    typename = matrix_type.name()
    if matrix_type is DenseMatrix:
      typename = "Matrix"
    elif matrix_type is SparseBuildMatrix:
      typename = "Matrix"
    else:
      raise RuntimeError, "an invalid matrix_type was specified"
  return makeRightType(matrices_internal, typename, typecode)




# construction functions ------------------------------------------------------
def array(data, typecode = None):
  # slow, but that doesn't matter so much
  def getRank(data):
    assert type(data) is types.ListType

    if data and type(data[0]) is types.ListType:
      return 1+getRank(data[0])
    else:
      return 1

  def getBiggestType(data, prev_biggest_type = Float64):
    assert type(data) is types.ListType

    if len(data) == 0:
      return biggest_type

    if type(data[0]) is types.ListType:
      for i in data:
        prev_biggest_type = getBiggestType(data, prev_biggest_type)
      return prev_biggest_type
    else:
      for i in data:
        if type(i) is types.ComplexType:
          prev_biggest_type = Complex64
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
  elif rank == 1:
    h = len(data)
    result = mat_class(h)
    for i in range(h):
      result[i] = data[i]



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




