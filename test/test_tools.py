import math
import cmath
import pylinear.matrices as num
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

def hermite(mat):
  return num.transpose(num.conjugate(mat))

def delta(x,y):
  if x == y:
    return 1
  else:
    return 0

  

def orthogonalizeInPlace(vectors):
  # Gram-Schmidt FIXME: unstable

  done_vectors = 0

  for v in vectors:
    for i in range(done_vectors):
      v -= sp(v,vectors[i]) * vectors[i]
    v_norm = norm2(v)
    if v_norm == 0:
      raise RuntimeError, "Orthogonalization failed"
    v /= v_norm
    done_vectors += 1


    

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
      print i,j,":",sp(vectors[i],vectors[j])
      assert abs(delta(i,j) - sp(vectors[i], vectors[j])) < 1e-12
  return vectors





def makeRandomOrthogonalMatrix(size, typecode):
  mat = num.zeros((size,size), typecode)
  vectors = []
  for i in range(size):
    v = mat[:,i]
    writeRandomVector(v)
    vectors.append(v)

  orthogonalizeInPlace(vectors)
  return mat


  

def makeRandomSPDMatrix(size, typecode):
  eigenvalues = makeRandomVector(size, typecode)
  eigenmat = num.zeros((size,size), num.Float)
  for i in range(size):
    eigenmat[i,i] = abs(eigenvalues[i])

  orthomat = makeRandomOrthogonalMatrix(size, typecode)
  return num.matrixmultiply(hermite(orthomat), num.matrixmultiply(eigenmat,orthomat))




def makeRandomMatrix(size, typecode, matrix_type = num.DenseMatrix):
  result = num.zeros((size, size), typecode)
  elements = size ** 2 / 10

  for i in range(elements):
    row = random.randrange(0, size)
    col = random.randrange(0, size)
    
    value = random.normalvariate(0,10)
    if typecode == num.Complex64:
      value += 1j*random.normalvariate(0,10)

    result[i,j] += value
  return result




  
def _test():
  print makeRandomSPDMatrix(4, num.Complex64)




if __name__ == "__main__":
  _test()
