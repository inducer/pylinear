import pylinear.matrices as num
import pylinear.algorithms as algo




def solve_linear_equations(mat, rhs):
  typecode = mat.typecode()
  h,w = mat.shape
  umf_operator = algo.makeUMFPACKMatrixOperator(
    num.asarray(mat, mat.typecode(), num.SparseExecuteMatrix))

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




def inverse(mat):
  w,h = mat.shape
  assert h == w
  return solve_linear_equations(mat, num.identity(h, mat.typecode()))




def determinant(mat):
  h,w = mat.shape
  assert h == w

  l,u, permut, sign = algo.lu(mat)

  product = 1
  for i in range(h):
    product *= u[i,i]

  return product * sign




def vandermonde(vector, degree = None):
  size, = vector.shape

  if degree is None:
    degree = size

  mat = num.array((size, size), vector.typecode())
  for i,v in zip(range(size), vector):
    for power in range(size):
      mat[i,power] = v**power
  return mat




def polyfit(x_vector, data_vector, degree):
  vdm = vandermonde(x_vector, degree)
  vdm2 = num.matrixmultiply(num.hermite(vdm), vdm)
  rhs = num.matrixmultiply(num.hermite, data_vector)
  return solve_linear_equations(vdm2, rhs) 

