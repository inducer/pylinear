import pylinear.matrices as num
from algorithms_internal import *



def _lookupClass(name_trunk, typecode, dict = None):
  name = name_trunk
  if typecode == num.Float64:
    name += "Float64"
  elif typecode == num.Complex64:
    name += "Complex64"
  else:
    raise RuntimeError, "Invalid element type requested"

  if dict is None:
    return eval(name)
  else:
    return dict[name]




def makeIdentityMatrixOperator(n, typecode):
  my_class = _lookupClass("IdentityMatrixOperator", typecode)
  return my_class(n)

def makeCGMatrixOperator(matrix_op, max_it, tol = 1e-12, precon_op = None):
  if precon_op is None:
    w,h = matrix_op.shape
    precon_op = makeIdentityMatrixOperator(w, matrix_op.typecode())

  my_class = _lookupClass("CGMatrixOperator", matrix_op.typecode())
  return my_class(matrix_op, precon_op, max_it, tol)
  


