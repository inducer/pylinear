import pylinear.matrices as num
import pylinear.algorithms as algo




def inverse(mat):
  typecode = mat.typecode()
  h,w = mat.shape
  umf_operator = algo.makeUMFPACKMatrixOperator(mat)
  inverse = num.zeros(mat.shape, typecode)

  temp = num.zeros((h,), typecode)
  for col in range(w):
    unit_vec = num.zeros((h,), typecode)
    unit_vec[col] = 1
    umf_operator.apply(unit_vec, temp)
    inverse[:,col] = temp
  return inverse




def determinant(mat):
  h,w = mat.shape
  assert h == w

  l,u = algo.lu(mat)

  product = 1
  for i in range(h):
    product *= u[i,i]

  return product

