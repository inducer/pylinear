import pylinear.matrices as num
import pylinear.algorithms as algo
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
  size = 1000
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


#elementary()
#cg()
umfpack()
  



