import pylinear.matrices as num
import pylinear.linear_algebra as la
import types




# polynomial fits -------------------------------------------------------------
def vandermonde(vector, degree = None):
  size, = vector.shape

  if degree is None:
    degree = size

  mat = num.zeros((size, degree), vector.typecode())
  for i,v in zip(range(size), vector):
    for power in range(degree):
      mat[i,power] = v**power
  return mat




def fit_polynomial(x_vector, data_vector, degree):
  vdm = vandermonde(x_vector, degree)
  vdm2 = num.matrixmultiply(num.hermite(vdm), vdm)
  rhs = num.matrixmultiply(num.hermite(vdm), data_vector)
  return la.solve_linear_equations(vdm2, rhs) 




def evaluate_polynomial(coefficients, x):
  # Horner evaluation
  size = len(coefficients)
  value_so_far = 0
  for i in range(size):
    value_so_far = x * value_so_far + coefficients[size-1-i]
  return value_so_far




def build_polynomial_function(coefficients):
  return lambda x: evaluate_polynomial(coefficients, x)




def approximate_polynomially(x_vector, f, degree):
  return fit_polynomial(x_vector, num.array(map(f, x_vector)), degree)
  



def get_approximant(x_vector, f, degree):
  return build_polynomial_function(approximate_polynomially(x_vector, f, degree))




# matlab-workalikes -----------------------------------------------------------
def linspace(x, y, n = 100):
  if type(x) is types.IntType:
    x = float(x)
  if type(y) is types.IntType:
    y = float(y)
  h = (y-x) / n
  return [ x+h*i for i in range(n+1) ]


