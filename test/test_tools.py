import math
import cmath
import pylinear.matrices as num
#import Numeric as num
#import numarray as num
import stopwatch
import random




def sp(x,y):
  return num.innerproduct(num.conjugate(y),x)




def norm2(x):
  scalar_product = sp(x,x)
  try:
    assert abs(scalar_product.imag) < 1e-12
    return math.sqrt(scalar_product.real)
  except AttributeError:
    # whoops. not complex
    return math.sqrt(scalar_product)

def delta(x,y):
  if x == y:
    return 1
  else:
    return 0

def hermite(x):
  return num.transpose(num.conjugate(x))

  

def orthogonalize(vectors):
  # Gram-Schmidt FIXME: unstable

  done_vectors = []

  for v in vectors:
    my_v = v.copy()
    for done_v in done_vectors:
      my_v -= sp(v,done_v) * done_v
    v_norm = norm2(my_v)
    if v_norm == 0:
      raise RuntimeError, "Orthogonalization failed"
    my_v /= v_norm
    done_vectors.append(my_v)
  return done_vectors


    

def writeRandomVector(vec):
  size, = vec.shape
  for i in range(size):
    value = random.normalvariate(0,10)
    if vec.typecode() == num.Complex64:
      value += 1j*random.normalvariate(0,10)
    vec[i] = value




def makeRandomVector(size, typecode):
  vec = num.zeros((size,), typecode)
  writeRandomVector(vec)
  return vec




def makeRandomONB(size, typecode):
  vectors = [ makeRandomVector(size, typecode) for i in range(size) ]
  orthogonalizeInPlace(vectors)

  for i in range(size):
    for j in range(size):
      assert abs(delta(i,j) - sp(vectors[i], vectors[j])) < 1e-12
  return vectors





def makeRandomOrthogonalMatrix(size, typecode):
  vectors = []
  for i in range(size):
    v = num.zeros((size,), typecode)
    writeRandomVector(v)
    vectors.append(v)

  orth_vectors = orthogonalize(vectors)

  mat = num.zeros((size,size), typecode)
  for i in range(size):
    mat[:,i] = orth_vectors[i]

  return mat


  

def makeRandomSPDMatrix(size, typecode):
  eigenvalues = makeRandomVector(size, typecode)
  eigenmat = num.zeros((size,size), typecode)
  for i in range(size):
    eigenmat[i,i] = abs(eigenvalues[i])

  orthomat = makeRandomOrthogonalMatrix(size, typecode)
  return num.matrixmultiply(hermite(orthomat), num.matrixmultiply(eigenmat,orthomat))




def makeFullRandomMatrix(size, typecode):
  result = num.zeros((size, size), typecode)

  for row in range(size):
    for col in range(size):
      value = random.normalvariate(0,10)
      if typecode == num.Complex64:
        value += 1j*random.normalvariate(0,10)

      result[row,col] = value
  return result




def makeRandomMatrix(size, typecode, matrix_type = num.DenseMatrix):
  result = num.zeros((size, size), typecode, matrix_type)
  elements = size ** 2 / 10

  for i in range(elements):
    row = random.randrange(0, size)
    col = random.randrange(0, size)
    
    value = random.normalvariate(0,10)
    if typecode == num.Complex64:
      value += 1j*random.normalvariate(0,10)

    result[row,col] += value
  return result




def makePermutationMatrix(permutation, typecode):
  size = len(permutation)
  result = num.zeros((size,size), typecode)
  for index, value in zip(range(size), permutation):
    result[index,value] = 1
  return result



  
def _test():
  size = 10
  job = stopwatch.tJob( "make spd" )
  for i in range(100):
    A = makeRandomSPDMatrix(size, num.Complex64)
  job.done()
  #print A




if __name__ == "__main__":
  _test()
