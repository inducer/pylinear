import pylinear.matrices as num
import pylinear.algorithms as algo




def solve_linear_equations(mat, rhs):
  typecode = mat.typecode()
  h,w = mat.shape
  umf_operator = algo.makeUMFPACKMatrixOperator(mat)

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

  l,u = algo.lu(mat)

  product = 1
  for i in range(h):
    product *= u[i,i]

  return product

