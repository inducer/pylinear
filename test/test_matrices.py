import sys, random
import pylinear.matrices as num
import pylinear.algorithms as algo
import pylinear.matrix_tools as mtools
import pylinear.linear_algebra as la
import pylinear.iteration as iteration
from test_tools import *
import unittest




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
        if len(matrix.shape) == 2:
          for i in matrix:
              for j in i:
                  self.assert_(j == 0)
        else:
            for j in matrix:
                self.assert_(j == 0)

    def doTestElementaryStuff(self, typecode):
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

        m = num.zeros((11,10), typecode)
        self.assertZero(m)

        v = num.zeros((11,), typecode)
        # believe you me, this test failed at one time
        # weird things happen :/
        self.assertZero(v)

    def testElementaryStuff(self):
        self.forAllTypecodes(self.doTestElementaryStuff)

    def doTestAddScattered(self,typecode):
        a = num.zeros((10,10), typecode)
        vec = num.array([3., 5.])
        b = num.asarray(num.outerproduct(vec, num.array([2., 4.])), typecode)
        a.addScattered([5,7], [1,3], b)

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
            v = num.zeros((2,), typecode)
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

    def doTestUmfpack(self, typecode):
        size = 100
        A = num.asarray(makeRandomMatrix(size, typecode), typecode, num.SparseExecuteMatrix)

        umf_op = algo.makeUMFPACKMatrixOperator(A)
        b = makeRandomVector(size, typecode)
        x = num.zeros((size,), typecode)

        umf_op.apply(b, x)

        self.assert_(norm2(b - num.matrixmultiply(A, x)) < 1e-10)

    def testUmfpack(self):
        self.forAllTypecodes(self.doTestUmfpack)

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
        L = la.cholesky_decomposition(A)
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
        size = 100
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

    def doTestJacobi(self, typecode):
        size = 10
        
        def off_diag_norm_squared(a):
            result = 0
            for i,j in a.indices():
                if i != j:
                    result += abs(a[i,j])**2
            return result

        a = mtools.makeRandomSPDMatrix(size, typecode)
        before = math.sqrt(off_diag_norm_squared(a))
        q, aprime = mtools.diagonalize(a, iteration.makeObserver(rel_goal = 1e-10))
        after = math.sqrt(off_diag_norm_squared(aprime))

        mm = num.matrixmultiply
        herm = num.hermite

        for i in range(size):
            evec = q[:,i]
            evalue = aprime[i,i]
            self.assert_(mtools.norm2(mm(a, evec) - evalue * evec) / mtools.norm2(evec) < 1e-8)

        self.assertSmall(mm(q, mm(aprime, herm(q))) - a)
        self.assert_(after / before <= 1e-10)


    def testJacobi(self):
        self.forAllTypecodes(self.doTestJacobi)

    def doTestCodiagonalization(self, typecode):
        size = 10
        
        def off_diag_norm_squared(a):
            result = 0
            for i,j in a.indices():
                if i != j:
                    result += abs(a[i,j])**2
            return result

        a = mtools.makeRandomSPDMatrix(size, typecode)
        before = math.sqrt(off_diag_norm_squared(a))
        q, mats_post, achieved = mtools.codiagonalize(
            [a], iteration.makeObserver(stall_thresh = 1e-5, rel_goal = 1e-10))
        aprime = mats_post[0]
        after = math.sqrt(off_diag_norm_squared(aprime))

        mm = num.matrixmultiply
        herm = num.hermite

        for i in range(size):
            evec = q[:,i]
            evalue = aprime[i,i]
            self.assert_(mtools.norm2(mm(a, evec) - evalue * evec) / mtools.norm2(evec) < 1e-8)

        self.assertSmall(mm(q, mm(aprime, herm(q))) - a)
        self.assert_(after / before <= 1e-10)

    def testCodiagonalization(self):
        self.forAllTypecodes(self.doTestCodiagonalization)

    def doTestMatrixExp(self, typecode):
        a = mtools.makeRandomSPDMatrix(20, num.Complex)
        e_a1 = mtools.matrixExp(a)
        e_a2 = mtools.matrixExpByDiagonalization(a)
        e_a3 = mtools.matrixExpBySymmetricDiagonalization(a)
        self.assert_(mtools.frobeniusNorm(e_a1-e_a2)
                     / mtools.frobeniusNorm(e_a1)
                     / mtools.frobeniusNorm(e_a2) <= 1e-15)
        self.assert_(mtools.frobeniusNorm(e_a1-e_a3)
                     / mtools.frobeniusNorm(e_a1)
                     / mtools.frobeniusNorm(e_a3) <= 1e-15)

    def testMatrixExp(self):
        self.forAllTypecodes(self.doTestMatrixExp)

    def doTestHeigenvectors(self, typecode):
        size = 100

        a = mtools.makeRandomSPDMatrix(size, typecode)
        q, w = la.Heigenvectors(a)
        w2 = la.Heigenvalues(a)

        self.assert_(abs(sum(w) - sum(w2)) < 1e-12)

        d = num.zeros(a.shape, a.typecode())
        for i in range(size):
            d[i,i] = w[i]
        mm = num.matrixmultiply
        self.assertSmall(a - mm(q, mm(d, num.hermite(q))))

    def testHeigenvectors(self):
        self.forAllTypecodes(self.doTestHeigenvectors)

    def doTestEigenvectors(self, typecode):
        size = 100

        a = mtools.makeFullRandomMatrix(size, typecode)
        evecs, evals = la.eigenvectors(a)
        evals2 = la.eigenvalues(a)

        self.assert_(abs(sum(evals) - sum(evals2)) < 1e-12)

        d = num.zeros(a.shape, num.Complex)
        for i in range(size):
            d[i,i] = evals[i]
        mm = num.matrixmultiply
        self.assertSmall(mm(a, evecs) - mm(evecs, d))

    def testEigenvectors(self):
        self.forAllTypecodes(self.doTestEigenvectors)

    def doTestBiCGSTAB(self, typecode):
        # real case fails sometimes
        size = 30

        A = makeFullRandomMatrix(size, typecode)
        b = makeRandomVector(size, typecode)

        A_op = algo.makeMatrixOperator(A)
        bicgstab_op = algo.makeBiCGSTABMatrixOperator(A_op, 40000, 1e-10)
        #bicgstab_op.debug_level = 1
        x = num.zeros((size,), typecode)

        initial_resid = norm2(b - num.matrixmultiply(A, x))
        bicgstab_op.apply(b, x)
        end_resid = norm2(b - num.matrixmultiply(A, x))
        #print typecode, end_resid/initial_resid 
        self.assert_(end_resid/initial_resid < 1e-10)

    def testBiCGSTAB(self):
        self.forAllTypecodes(self.doTestBiCGSTAB)

    def testComplexAdaptor(self):
        size = 40

        a = makeFullRandomMatrix(size, num.Complex)
        a_op = algo.makeMatrixOperator(a)
        a2_op = algo.adaptRealToComplexOperator(
            algo.makeMatrixOperator(a.real), 
            algo.makeMatrixOperator(a.imaginary))

        for i in range(20):
            b = makeRandomVector(size, num.Complex)
            result1 = num.zeros((size,), num.Complex)
            result2 = num.zeros((size,), num.Complex)
            a_op.apply(b, result1)
            a2_op.apply(b, result2)
            self.assert_(mtools.norm2(result1 - result2) < 1e-11)

    def doTestInterpolation(self, typecode):
        size = 4

        def eval_at(x):
            # Horner evaluation
            result = 0.
            for i in coefficients[::-1]:
                result = i + x*result
            return result

        abscissae = makeRandomVector(size, typecode)
        coefficients = makeRandomVector(size, typecode)
        values = num.array([eval_at(abscissa) for abscissa in abscissae])

        for i in range(10):
            to_x = random.normalvariate(0,100)
            i_coeff = mtools.findInterpolationCoefficients(abscissae, to_x)
            f_x1 = eval_at(to_x)

            self.assert_(abs(num.innerproduct(i_coeff, values)-f_x1) < 1e-7)

    def testInterpolation(self):
        self.forAllTypecodes(self.doTestInterpolation)

            
if __name__ == '__main__':
    unittest.main()
