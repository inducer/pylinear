import sys
import pylinear.matrices as num
import pylinear.algorithms as algo
import pylinear.linear_algebra as la
from test_tools import *

def elementary():
  mat = num.zeros((3, 3), num.Complex64)
  mat[1,2] += 5+3j
  mat[2,1] += 7-8j
  #print num.hermite(mat)[1]
  #print num.hermite(mat)[1,:]

  vec = num.zeros((3,), num.Complex64)
  for i in vec.indices():
    print i
    vec[i] = 17
  mat[0] = vec

  for i in mat:
    print i
  for i in mat.indices():
    print i, mat[i]

  for i in vec:
    print i
  print sum(vec)
  print num.matrixmultiply(mat, mat)

def addScattered(typecode):
  a = num.zeros((10,10), typecode)
  vec = num.array([3., 5.])
  b = num.asarray(num.outerproduct(vec, num.array([2., 4.])), typecode)
  a.addScattered([5,7], [1,3], b)
  print a

def broadcast(typecode):
  size = 10
  a = makeFullRandomMatrix(size, typecode)

  def assertZero(matrix):
    for i in matrix:
      for j in i:
        assert j == 0

  def scalar_broadcast(a):
    a[3:7, 5:9] = 0
    assertZero(a[3:7, 5:9])

  def scalar_broadcast2(a):
    a[3:7] = 0
    assertZero(a[3:7])

  def vec_broadcast(a):
    v = num.zeros((size,), typecode)
    a[3:7] = v
    assertZero(a[3:7])

  def vec_broadcast2(a):
    v = num.zeros((size,), typecode)
    a[:,2:4] = v
    assertZero(a[:, 2:4])
    
  scalar_broadcast(a.copy())
  scalar_broadcast2(a.copy())
  vec_broadcast(a.copy())
  vec_broadcast2(a.copy())

def ufunc():
  vec = num.array([3., 5.])
  a = num.outerproduct(vec, num.array([2., 4.]))

  vec2 = num.array([1., 17.])
  b = num.outerproduct(vec2, num.array([5., 1.]))

  print "a:", a
  print "b:", b
  print "min(a,b):", num.minimum(a,b)
  print "max(a,b):", num.maximum(a,b)

  print "a*2 residual:", num.multiply(a, 2) - 2 * a
  print "a*vec:", num.multiply(a, vec)
  print "a*a:", num.multiply(a, a)
  print "2*a residual:", num.multiply(a, 2) - num.multiply(2, a)
  print "vec*a residual:", num.multiply(a, vec) - num.multiply(vec, a)

def cg(typecode):
  size = 100

  job = stopwatch.tJob( "make spd" )
  A = makeRandomSPDMatrix(size, typecode)
  Aop = algo.makeMatrixOperator(A)
  b = makeRandomVector(size, typecode)
  cg_op = algo.makeCGMatrixOperator(Aop, 1000)
  x = num.zeros((size,), typecode)
  job.done()

  job = stopwatch.tJob( "cg" )
  cg_op.apply(b, x)
  job.done()

  print norm2(b - num.matrixmultiply(A, x))

def umfpack(typecode):
  size = 100
  job = stopwatch.tJob("make matrix")
  A = num.asarray(makeRandomMatrix(size, typecode), typecode, num.SparseExecuteMatrix)
  job.done()

  job = stopwatch.tJob("umfpack")
  umf_op = algo.makeUMFPACKMatrixOperator(A)
  job.done()
  b = makeRandomVector(size, typecode)
  x = num.zeros((size,), typecode)

  umf_op.apply(b, x)

  print x

  print norm2(b - num.matrixmultiply(A, x))

class tMyMatrixOperator(algo.MatrixOperatorFloat64):
  def __init__(self, mat):
    self.Matrix = mat

  def typecode(self):
    return num.Float64

  def size(self):
    w,h = self.Matrix.shape
    return w

  def apply(self, before, after):
    after[:] = num.matrximultiply(self.Matrix, after)

def matrixoperator():
  size = 100

  job = stopwatch.tJob( "make spd" )
  A = makeRandomSPDMatrix(size, num.Float64)
  Aop = tMyMatrixOperator(A)
  b = makeRandomVector(size, num.Float64)
  cg_op = algo.makeCGMatrixOperator(Aop, 1000)
  x = num.zeros((size,), num.Float64)
  job.done()

  job = stopwatch.tJob( "cg" )
  cg_op.apply(b, x)
  job.done()

  print norm2(b - num.matrixmultiply(A, x))

  
def arpack_generalized(typecode):
  size = 200
  A = makeRandomMatrix(size, typecode)
  Aop = algo.makeMatrixOperator(A)
  #Iop = algo.makeIdentityMatrixOperator(size, typecode)

  job = stopwatch.tJob("make spd")
  M = num.asarray(makeRandomSPDMatrix(size, typecode), typecode,
    num.SparseExecuteMatrix)
  job.done()

  Mop = algo.makeMatrixOperator(M)

  job = stopwatch.tJob( "umfpack factor")
  Minvop = algo.makeUMFPACKMatrixOperator(M)
  job.done()

  OP = algo.composeMatrixOperators(Minvop, Aop)

  job = stopwatch.tJob("arpack rci")
  results = algo.runArpack(OP, Mop, algo.REGULAR_GENERALIZED,
    0, 5, 10, algo.LARGEST_MAGNITUDE, 1e-12, False, 0)
  job.done()

  Acomplex = num.asarray(A, num.Complex)
  Mcomplex = num.asarray(M, num.Complex)
  for value,vector in zip(results.RitzValues, results.RitzVectors):
    print "eigenvalue:", value
    #print "eigenvector:", vector
    print "residual:", norm2(num.matrixmultiply(Acomplex,vector) - value * 
      num.matrixmultiply(Mcomplex, vector))
    print

def arpack_shift_invert(typecode):
  size = 200
  sigma = 1

  A = makeRandomMatrix(size, typecode)

  job = stopwatch.tJob("make spd")
  M = makeRandomSPDMatrix(size, typecode)
  job.done()

  Mop = algo.makeMatrixOperator(M)

  job = stopwatch.tJob( "shifted matrix")
  shifted_mat = num.asarray(A - sigma * M, typecode, num.SparseExecuteMatrix)
  job.done()

  job = stopwatch.tJob( "umfpack factor")
  shifted_mat_invop = algo.makeUMFPACKMatrixOperator(shifted_mat)
  job.done()

  OP = algo.composeMatrixOperators(shifted_mat_invop, Mop)

  job = stopwatch.tJob("arpack rci")
  results = algo.runArpack(OP, Mop, algo.SHIFT_AND_INVERT_GENERALIZED,
    sigma, 5, 10, algo.LARGEST_MAGNITUDE, 1e-12, False, 0)
  job.done()

  Acomplex = num.asarray(A, num.Complex)
  Mcomplex = num.asarray(M, num.Complex)
  for value,vector in zip(results.RitzValues, results.RitzVectors):
    print "eigenvalue:", value
    #print "eigenvector:", vector
    print "residual:", norm2(num.matrixmultiply(Acomplex,vector) - value * 
      num.matrixmultiply(Mcomplex, vector))
    print

def getResidual(matrix):
  my_sum = 0
  for i in matrix:
    my_sum += num.innerproduct(num.conjugate(i),i)
  return my_sum

def cholesky(typecode):
  size = 100
  A = makeRandomSPDMatrix(size, typecode)
  L = algo.cholesky(A)
  resid = num.matrixmultiply(L,hermite(L))-A
  print "cholesky residual:", getResidual(resid)

def lu(typecode):
  size = 500
  A = makeFullRandomMatrix(size, typecode)
  job = stopwatch.tJob("lu")
  L,U,permut,sign = algo.lu(A)
  job.done()
  print permut
  permut_mat = makePermutationMatrix(permut, typecode)
  permut_a = num.matrixmultiply(permut_mat, A)
  print "lu residual:", getResidual(num.matrixmultiply(L, U)-permut_a)
  #print num.matrixmultiply(L, U)-permut_a

def sparse(typecode):
  def countElements(mat):
    count = 0
    for i in mat.indices():
      count += 1
    return count

  size = 100
  A1 = makeRandomMatrix(size, typecode, num.SparseBuildMatrix)
  A2 = num.asarray(A1, typecode, num.SparseExecuteMatrix)
  print "sparse:", countElements(A1), countElements(A2)

def inverse(typecode):
  size = 100
  A = makeFullRandomMatrix(size, typecode)
  Ainv = la.inverse(A)
  Id = num.identity(size, typecode)

  print "inverse residual 1:", getResidual(num.matrixmultiply(Ainv,A)-Id)
  print "inverse residual 2:", getResidual(num.matrixmultiply(A,Ainv)-Id)

def determinant(typecode):
  size = 10
  A = makeFullRandomMatrix(size, typecode)
  detA = la.determinant(A)
  A2 = num.matrixmultiply(A, A)
  detA2 = la.determinant(A2)

  print "determinant:", abs((detA**2-detA2) / detA2)



def testAll(typecode):
  print "elementary:"
  elementary()
  print "-------------------------------------"
  print "addScattered:"
  addScattered(typecode)
  print "-------------------------------------"
  print "broadcast:"
  broadcast(num.Float)
  print "-------------------------------------"
  print "ufunc:"
  ufunc()
  print "-------------------------------------"
  print "cg:"
  cg(typecode)
  print "-------------------------------------"
  print "umfpack:"
  umfpack(typecode)
  print "-------------------------------------"
  # pending fix from BPL gurus.
  #matrixoperator()
  print "-------------------------------------"
  print "arpack:"
  arpack_generalized(typecode)
  print "-------------------------------------"
  print "arpack_shift_invert:"
  arpack_shift_invert(typecode)
  print "-------------------------------------"
  print "cholesky:"
  cholesky(typecode)
  print "-------------------------------------"
  print "lu:"
  lu(typecode)
  print "-------------------------------------"
  print "sparse:"
  sparse(typecode)
  print "-------------------------------------"
  print "inverse:"
  inverse(typecode)
  print "-------------------------------------"
  print "determinant:"
  determinant(typecode)




testAll(num.Float)
print "-------------------------------------"
testAll(num.Complex)
