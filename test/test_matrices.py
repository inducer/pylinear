import sys
import pylinear.matrices as num
import pylinear.algorithms as algo
import pylinear.matrix_tools as mtools
import pylinear.linear_algebra as la
import unittest
from test_tools import *




# TEST ME!
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




class tTestMatrices(unittest.TestCase):
    def forAllTypecodes(self, f):
        f(num.Float)
        f(num.Complex)

    def assertSmall(self, matrix):
        self.assert_(mtools.frobeniusNorm(matrix) < 1e-10)

    def assertZero(self, matrix):
        for i in matrix:
            for j in i:
                self.assert_(j == 0)

    def testElementaryStuff(self):
        mat = num.zeros((3, 3), num.Complex64)
        mat[1,2] += 5+3j
        mat[2,1] += 7-8j

        #print num.hermite(mat)[1]
        #print num.hermite(mat)[1,:]

        vec = num.zeros((3,), num.Complex64)
        for i in vec.indices():
            vec[i] = 17
        mat[0] = vec

        h,w = mat.shape
        count = 0
        for i in mat:
            count += 1
        self.assert_(count == h)

        count = 0
        for i in mat.indices():
            count += 1
        self.assert_(count == h*w)

        sum(vec)
        num.matrixmultiply(mat, mat)

    def doTestAddScattered(self,typecode):
        a = num.zeros((10,10), typecode)
        vec = num.array([3., 5.])
        b = num.asarray(num.outerproduct(vec, num.array([2., 4.])), typecode)
        a.addScattered([5,7], [1,3], b)
        #print a

    def testAddScattered(self):
        self.forAllTypecodes(self.doTestAddScattered)

    def doTestBroadcast(self, typecode):
        size = 10
        a = makeFullRandomMatrix(size, typecode)

        def scalar_broadcast(a):
            a[3:7, 5:9] = 0
            self.assertZero(a[3:7, 5:9])

        def scalar_broadcast2(a):
            a[3:7] = 0
            self.assertZero(a[3:7])

        def vec_broadcast(a):
            v = num.zeros((size,), typecode)
            a[3:7] = v
            self.assertZero(a[3:7])

        def vec_broadcast2(a):
            v = num.zeros((size,), typecode)
            a[:,2:4] = v
            self.assertZero(a[:, 2:4])

        scalar_broadcast(a.copy())
        scalar_broadcast2(a.copy())
        vec_broadcast(a.copy())
        vec_broadcast2(a.copy())

    def testBroadcast(self):
        self.forAllTypecodes(self.doTestBroadcast)

    def testUfunc(self):
        vec = num.array([3., 5.])
        a = num.outerproduct(vec, num.array([2., 4.]))

        vec2 = num.array([1., 17.])
        b = num.outerproduct(vec2, num.array([5., 1.]))

        minab = num.minimum(a,b)
        maxab = num.maximum(a,b)

        self.assertSmall(num.multiply(a, 2) - 2 * a)
        num.multiply(a, a)
        self.assertSmall(num.multiply(a, 2) - num.multiply(2, a))
        self.assertSmall(num.multiply(a, vec) - num.multiply(vec, a))

    def doTestCG(self, typecode):
        size = 100

        A = makeRandomSPDMatrix(size, typecode)
        Aop = algo.makeMatrixOperator(A)
        b = makeRandomVector(size, typecode)
        cg_op = algo.makeCGMatrixOperator(Aop, 4000, 1e-10)
        x = num.zeros((size,), typecode)

        initial_resid = norm2(b - num.matrixmultiply(A, x))
        cg_op.apply(b, x)
        end_resid = norm2(b - num.matrixmultiply(A, x))
        self.assert_(end_resid/initial_resid < 1e-10)

    def testCG(self):
        self.forAllTypecodes(self.doTestCG)

    def umfpack(typecode):
        size = 100
        A = num.asarray(makeRandomMatrix(size, typecode), typecode, num.SparseExecuteMatrix)

        umf_op = algo.makeUMFPACKMatrixOperator(A)
        b = makeRandomVector(size, typecode)
        x = num.zeros((size,), typecode)

        umf_op.apply(b, x)

        self.assert_(norm2(b - num.matrixmultiply(A, x)) < 1e-10)

    def doTestArpackGeneralized(self, typecode):
        size = 100
        A = makeRandomMatrix(size, typecode)
        Aop = algo.makeMatrixOperator(A)

        M = num.asarray(makeRandomSPDMatrix(size, typecode), typecode,
                        num.SparseExecuteMatrix)

        Mop = algo.makeMatrixOperator(M)

        Minvop = algo.makeUMFPACKMatrixOperator(M)

        OP = algo.composeMatrixOperators(Minvop, Aop)

        results = algo.runArpack(OP, Mop, algo.REGULAR_GENERALIZED,
                                 0, 5, 10, algo.LARGEST_MAGNITUDE, 1e-12, False, 0)

        Acomplex = num.asarray(A, num.Complex)
        Mcomplex = num.asarray(M, num.Complex)
        for value,vector in zip(results.RitzValues, results.RitzVectors):
            self.assert_(norm2(num.matrixmultiply(Acomplex,vector) - value * 
                              num.matrixmultiply(Mcomplex, vector)) < 1e-7)

    def testArpackGeneralized(self):
        self.forAllTypecodes(self.doTestArpackGeneralized)

    def doTestArpackShiftInvert(self, typecode):
        size = 100
        sigma = 1

        A = makeRandomMatrix(size, typecode)

        M = makeRandomSPDMatrix(size, typecode)
        Mop = algo.makeMatrixOperator(M)

        shifted_mat = num.asarray(A - sigma * M, typecode, num.SparseExecuteMatrix)

        shifted_mat_invop = algo.makeUMFPACKMatrixOperator(shifted_mat)

        OP = algo.composeMatrixOperators(shifted_mat_invop, Mop)

        results = algo.runArpack(OP, Mop, algo.SHIFT_AND_INVERT_GENERALIZED,
                                 sigma, 5, 10, algo.LARGEST_MAGNITUDE, 1e-12, False, 0)

        Acomplex = num.asarray(A, num.Complex)
        Mcomplex = num.asarray(M, num.Complex)
        for value,vector in zip(results.RitzValues, results.RitzVectors):
            self.assert_( norm2(num.matrixmultiply(Acomplex,vector) - value * 
                                num.matrixmultiply(Mcomplex, vector)) < 1e-10)

    def testArpackShiftInvert(self):
        self.forAllTypecodes(self.doTestArpackShiftInvert)

    def doTestCholesky(self, typecode):
        size = 100
        A = makeRandomSPDMatrix(size, typecode)
        L = algo.cholesky(A)
        self.assertSmall(num.matrixmultiply(L,hermite(L))-A)

    def testCholesky(self):
        self.forAllTypecodes(self.doTestCholesky)

    def doTestSolve(self, typecode):
        size = 200
        A = makeFullRandomMatrix(size, typecode)
        b = num.zeros((size,), typecode)
        writeRandomVector(b)
        x = la.solve_linear_equations(A,b)
        self.assert_(norm2(num.matrixmultiply(A,x)-b) < 1e-10)

    def testSolve(self):
        self.forAllTypecodes(self.doTestSolve)

    def doTestLU(self, typecode):
        size = 300
        A = makeFullRandomMatrix(size, typecode)
        L,U,permut,sign = algo.lu(A)
        permut_mat = makePermutationMatrix(permut, typecode)
        permut_a = num.matrixmultiply(permut_mat, A)
        self.assertSmall(num.matrixmultiply(L, U)-permut_a)

    def testLU(self):
        self.forAllTypecodes(self.doTestLU)

    def doTestSparse(self, typecode):
        def countElements(mat):
            count = 0
            for i in mat.indices():
                count += 1
            return count

        size = 100
        A1 = makeRandomMatrix(size, typecode, num.SparseBuildMatrix)
        A2 = num.asarray(A1, typecode, num.SparseExecuteMatrix)
        self.assert_(countElements(A1) == countElements(A2))

    def testSparse(self):
        self.forAllTypecodes(self.doTestSparse)

    def doTestInverse(self,typecode):
        size = 100
        A = makeFullRandomMatrix(size, typecode)
        Ainv = la.inverse(A)
        Id = num.identity(size, typecode)

        self.assertSmall(num.matrixmultiply(Ainv,A)-Id)
        self.assertSmall(num.matrixmultiply(A,Ainv)-Id)

    def testInverse(self):
        self.forAllTypecodes(self.doTestInverse)

    def doTestDeterminant(self, typecode):
        size = 10
        A = makeFullRandomMatrix(size, typecode)
        detA = la.determinant(A)
        A2 = num.matrixmultiply(A, A)
        detA2 = la.determinant(A2)

        self.assert_(abs((detA**2-detA2) / detA2) < 1e-10)

    def testDeterminant(self):
        self.forAllTypecodes(self.doTestDeterminant)

    def doTestSVD(self, typecode):
        size = 100
        mat = mtools.makeFullRandomMatrix(size, num.Complex)
        u, s_vec, vt = la.singular_value_decomposition(mat)
        
        s = num.zeros(mat.shape, s_vec.typecode())
        for i, v in enumerate(s_vec):
            s[i,i] = v

        mm = num.matrixmultiply
        mat_prime = mm(u, mm(s, vt))
        self.assertSmall(mat_prime - mat)

    def testSVD(self):
        self.forAllTypecodes(self.doTestSVD)




if __name__ == '__main__':
    unittest.main()
