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
    vec[i] = 17
  mat[0] = vec

  for i in mat:
    print i
  for i in mat.indices():
    print i, mat[i]

  print sum(vec)
  print num.matrixmultiply(mat, mat)

def cg():
  size = 100

  job = stopwatch.tJob( "make spd" )
  A = makeRandomSPDMatrix(size, num.Complex64)
  Aop = algo.makeMatrixOperator(A)
  b = makeRandomVector(size, num.Complex64)
  cg_op = algo.makeCGMatrixOperator(Aop, 1000)
  x = num.zeros((size,), num.Complex64)
  job.done()

  job = stopwatch.tJob( "cg" )
  cg_op.apply(b, x)
  job.done()

  print norm2(b - num.matrixmultiply(A, x))

def umfpack():
  size = 100
  job = stopwatch.tJob("make matrix")
  A = makeRandomMatrix(size, num.Complex)
  job.done()

  job = stopwatch.tJob("umfpack")
  umf_op = algo.makeUMFPACKMatrixOperator(A)
  job.done()
  b = makeRandomVector(size, num.Complex64)
  x = num.zeros((size,), num.Complex64)

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
  M = makeRandomSPDMatrix(size, typecode)
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
  shifted_mat = A - sigma * M
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

def sumAbsoluteValues(matrix):
  my_sum = 0
  for i in matrix:
    my_sum += abs(i)
  return my_sum

def cholesky(typecode):
  size = 100
  A = makeRandomSPDMatrix(size, typecode)
  L = algo.cholesky(A)
  resid = num.matrixmultiply(L,hermite(L))-A
  print "cholesky residual:", sumAbsoluteValues(resid)

def lu(typecode):
  size = 100
  A = makeFullRandomMatrix(size, typecode)
  L,U = algo.lu(A)
  print "lu residual:", sumAbsoluteValues(num.matrixmultiply(L,U)-A)

def sparse(typecode):
  def countElements(mat):
    count = 0
    for i in mat.indices():
      count += 1
    return count

  size = 100
  A1 = makeRandomMatrix(100, typecode, num.SparseBuildMatrix)
  A2 = num.asarray(A1, typecode, num.SparseExecuteMatrix)
  print "sparse:", countElements(A1), countElements(A2)

def inverse(typecode):
  size = 100
  A = makeFullRandomMatrix(size, typecode)
  Ainv = la.inverse(A)
  Id = num.identity(size, typecode)

  print "inverse residual 1:", sumAbsoluteValues(num.matrixmultiply(Ainv,A)-Id)
  print "inverse residual 2:", sumAbsoluteValues(num.matrixmultiply(A,Ainv)-Id)

def determinant(typecode):
  size = 10
  A = makeFullRandomMatrix(size, typecode)
  detA = la.determinant(A)
  A2 = num.matrixmultiply(A, A)
  detA2 = la.determinant(A2)

  print "determinant:", abs((detA**2-detA2) / detA2)




elementary()
print "-------------------------------------"
cg()
print "-------------------------------------"
umfpack()
print "-------------------------------------"
# pending fix from BPL gurus.
#matrixoperator()
print "-------------------------------------"
arpack_generalized(num.Complex)
print "-------------------------------------"
arpack_shift_invert(num.Float)
print "-------------------------------------"
cholesky(num.Complex)
print "-------------------------------------"
lu(num.Complex)
print "-------------------------------------"
sparse(num.Complex)
print "-------------------------------------"
inverse(num.Complex)
print "-------------------------------------"
determinant(num.Complex)


