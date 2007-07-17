import sys, random, math
import pylinear.array as num
import pylinear.operator as op
import pylinear.computation as comp
from pylinear.randomized import *
import pylinear.linear_algebra as la
import pylinear.toybox as toybox
import pylinear.iteration as iteration
import unittest
import test_matrices_data as tmd




class TestMatrices(unittest.TestCase):
    def for_all_dtypes(self, f):
        f(num.Float)
        f(num.Complex)

    def assert_small(self, matrix):
        self.assert_(comp.norm_frobenius(matrix) < 1e-10)

    def assert_zero(self, matrix):
        if len(matrix.shape) == 2:
          for i in matrix:
              for j in i:
                  self.assert_(j == 0)
        else:
            for j in matrix:
                self.assert_(j == 0)

    def do_test_elementary_stuff(self, dtype):
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
        mat * mat

        m = num.zeros((11,10), dtype)
        self.assert_zero(m)

        v = num.zeros((11,), dtype)

        # believe you me, this test failed at one time.
        # weird things happen :/
        self.assert_zero(v)

    def test_elementary_stuff(self):
        self.for_all_dtypes(self.do_test_elementary_stuff)

    def do_test_add_scattered(self,dtype):
        a = num.zeros((10,10), dtype)
        vec = num.array([3., 5.])
        b = num.asarray(vec <<num.outer>> num.array([2., 4.]), dtype)
        a.add_scattered([5,7], [1,3], b)

    def test_add_scattered(self):
        self.for_all_dtypes(self.do_test_add_scattered)

    def do_test_broadcast(self, dtype):
        size = 10
        a = make_random_full_matrix(size, dtype)

        def scalar_broadcast(a):
            a[3:7, 5:9] = 0
            self.assert_zero(a[3:7, 5:9])

        def scalar_broadcast2(a):
            a[3:7] = 0
            self.assert_zero(a[3:7])

        def vec_broadcast(a):
            v = num.zeros((size,), dtype)
            a[3:7] = v
            self.assert_zero(a[3:7])

        def vec_broadcast2(a):
            v = num.zeros((2,), dtype)
            a[:,2:4] = v
            self.assert_zero(a[:, 2:4])

        scalar_broadcast(a.copy())
        scalar_broadcast2(a.copy())
        vec_broadcast(a.copy())
        vec_broadcast2(a.copy())

    def test_broadcast(self):
        self.for_all_dtypes(self.do_test_broadcast)

    def test_ufunc(self):
        vec = num.array([3., 5.])
        a = vec <<num.outer>> num.array([2., 4.])

        vec2 = num.array([1., 17.])
        b = vec2 <<num.outer>> num.array([5., 1.])

        minab = num.minimum(a, b)
        maxab = num.maximum(a, b)

        self.assert_small(num.multiply(a, 2) - 2 * a)
        num.multiply(a, a)
        self.assert_small(num.multiply(a, 2) - num.multiply(2, a))
        self.assert_small(num.multiply(a, vec) - num.multiply(vec, a))

    def do_test_cg(self, dtype):
        size = 20

        A = make_random_spd_matrix(size, dtype)
        Aop = op.MatrixOperator.make(A)
        b = make_random_vector(size, dtype)
        cg_op = op.CGOperator.make(Aop, 4000, 1e-10)
        x = num.zeros((size,), dtype)

        initial_resid = comp.norm_2(b - A*x)
        x = cg_op(b)
        end_resid = comp.norm_2(b - A* x)
        self.assert_(end_resid/initial_resid < 1e-10)

    def test_cg(self):
        self.for_all_dtypes(self.do_test_cg)

    def do_test_umfpack(self, dtype):
        size = 100
        #A = make_random_matrix(size, dtype, num.SparseExecuteMatrix)
        #b = make_random_vector(size, dtype)
        A = tmd.umf_a[dtype]
        b = tmd.umf_b[dtype]

        umf_op = op.UMFPACKOperator.make(A)
        x = num.zeros((size,), dtype)

        umf_op.apply(b, x)

        self.assert_(comp.norm_2(b - A * x) < 1e-10)

    def test_umfpack(self):
        self.for_all_dtypes(self.do_test_umfpack)

    def do_test_arpack_classic(self, dtype):
        size = 10
        #A = make_random_matrix(size, dtype)
        A = tmd.aclassmat[dtype]
        Aop = op.MatrixOperator.make(A)

        results = comp.operator_eigenvectors(Aop, 3)

        for value,vector in results:
            self.assert_(comp.norm_2(A*vector - value*vector) < 1e-7)

    def test_arpack_classic(self):
        self.for_all_dtypes(self.do_test_arpack_classic)

    def do_test_arpack_generalized(self, dtype):
        size = 100
        A = tmd.agen_a[dtype]
        Aop = op.MatrixOperator.make(A)

        M = tmd.agen_m[dtype]
        Mop = op.MatrixOperator.make(M)

        Minvop = op.LUInverseOperator.make(M)

        results = comp.operator_eigenvectors(Minvop*Aop, 5, Mop)

        for value,vector in results:
            self.assert_(comp.norm_2(A*vector - value * 
                                   M*vector) < 1e-7)

    def test_arpack_generalized(self):
        self.for_all_dtypes(self.do_test_arpack_generalized)

    def do_test_arpack_shift_invert(self, dtype):
        size = 100
        sigma = 1

        #A = make_random_matrix(size, dtype)
        #M = make_random_spd_matrix(size, dtype)
        A = tmd.arpsi_a[dtype]
        M = tmd.arpsi_m[dtype]
        Mop = op.MatrixOperator.make(M)

        shifted_mat_invop = op.LUInverseOperator.make(A - sigma * M)

        results = comp.operator_eigenvectors(
            shifted_mat_invop * Mop, 5, Mop, spectral_shift=sigma)

        for value,vector in results:
            self.assert_(comp.norm_2(A*vector - value*M*vector) < 1e-10)

    def test_arpack_shift_invert(self):
        self.for_all_dtypes(self.do_test_arpack_shift_invert)

    def do_test_cholesky(self, dtype):
        size = 100
        A = make_random_spd_matrix(size, dtype)
        L = comp.cholesky(A)
        self.assert_small(L*L.H-A)

    def test_cholesky(self):
        self.for_all_dtypes(self.do_test_cholesky)

    def do_test_solve(self, dtype):
        size = 200
        #A = make_random_full_matrix(size, dtype)
        #b = num.zeros((size,), dtype)
        #write_random_vector(b)
        A = tmd.solve_a[dtype]
        b = tmd.solve_b[dtype]
        x = A <<num.solve>> b
        self.assert_(comp.norm_2(A*x - b) < 1e-10)

    def test_solve(self):
        self.for_all_dtypes(self.do_test_solve)

    def do_test_lu(self, dtype):
        size = 100
        A = make_random_full_matrix(size, dtype)+10*num.identity(size, num.Float)
        L,U,permut,sign = comp.lu(A)
        permut_mat = num.permutation_matrix(from_indices=permut)
        self.assert_small(L * U - permut_mat * A)

        inv_op = op.LUInverseOperator.make(A)
        for count in range(20):
            x = make_random_vector(size, dtype)
            self.assert_small(inv_op(A*x) - x)

    def test_lu(self):
        #self.for_all_dtypes(self.do_test_lu)
        self.do_test_lu(num.Float)

    def do_test_sparse(self, dtype):
        def count_elements(mat):
            count = 0
            for i in mat.indices():
                count += 1
            return count

        size = 100
        A1 = make_random_matrix(size, dtype, num.SparseBuildMatrix)
        A2 = num.asarray(A1, dtype, num.SparseExecuteMatrix)
        self.assert_(count_elements(A1) == count_elements(A2))

    def test_sparse(self):
        self.for_all_dtypes(self.do_test_sparse)

    def do_test_inverse(self,dtype):
        size = 100
        A = make_random_full_matrix(size, dtype)
        Ainv = la.inverse(A)
        Id = num.identity(size, dtype)

        self.assert_small(Ainv*A-Id)
        self.assert_small(A*Ainv-Id)

    def test_inverse(self):
        self.for_all_dtypes(self.do_test_inverse)

    def do_test_determinant(self, dtype):
        size = 10
        A = make_random_full_matrix(size, dtype)
        detA = comp.determinant(A)
        A2 = A*A
        detA2 = comp.determinant(A2)

        self.assert_(abs((detA**2-detA2) / detA2) < 1e-10)

    def test_determinant(self):
        self.for_all_dtypes(self.do_test_determinant)

    def do_test_svd(self, dtype):
        size = 100
        mat = make_random_full_matrix(size, dtype)
        u, s_vec, vt = comp.svd(mat)
        self.assert_small(u * num.diagonal_matrix(s_vec) * vt - mat)

    def test_svd(self):
        self.for_all_dtypes(self.do_test_svd)

    def do_test_pseudo_inverse(self, dtype):
        size = 100
        mat = make_random_full_matrix(size, dtype)[0:size,0:size-17]
        pi = comp.pseudo_inverse(mat)
        resid = pi * mat * pi - pi
        self.assert_small(resid)

    def test_pseudo_inverse(self):
        self.for_all_dtypes(self.do_test_pseudo_inverse)

    def do_test_jacobi(self, dtype):
        size = 10
        
        def off_diag_norm_squared(a):
            result = 0
            for i,j in a.indices():
                if i != j:
                    result += abs(a[i,j])**2
            return result

        #a = make_random_spd_matrix(size, dtype)
        a = tmd.jacspd[dtype]
        before = math.sqrt(off_diag_norm_squared(a))
        q, aprime = toybox.diagonalize_jacobi(a, iteration.make_observer(rel_goal = 1e-10))
        after = math.sqrt(off_diag_norm_squared(aprime))

        for i in range(size):
            evec = q[:,i]
            evalue = aprime[i,i]
            self.assert_(comp.norm_2(a*evec - evalue * evec) / comp.norm_2(evec) < 1e-8)

        self.assert_small(q * aprime * q.H - a)
        self.assert_(after / before <= 1e-10)


    def test_jacobi(self):
        self.for_all_dtypes(self.do_test_jacobi)

    def do_test_codiagonalization(self, dtype):
        size = 10
        
        def off_diag_norm_squared(a):
            return comp.norm_frobenius_squared(a) - comp.norm_2_squared(num.diagonal(a))

        a = make_random_spd_matrix(size, dtype)
        before = math.sqrt(off_diag_norm_squared(a))
        q, mats_post, achieved = toybox.codiagonalize(
            [a], iteration.make_observer(stall_thresh = 1e-5, rel_goal = 1e-10))
        aprime = mats_post[0]
        after = math.sqrt(off_diag_norm_squared(aprime))

        for i in range(size):
            evec = q[:,i]
            evalue = aprime[i,i]
            self.assert_(comp.norm_2(a*evec - evalue * evec) / comp.norm_2(evec) < 1e-8)

        self.assert_small(q * aprime * q.H - a)
        self.assert_(after / before <= 1e-10)

    def test_codiagonalization(self):
        self.for_all_dtypes(self.do_test_codiagonalization)

    def do_test_matrix_exp(self, dtype):
        a = make_random_spd_matrix(20, num.Complex)
        e_a1 = toybox.matrix_exp_by_series(a)
        e_a2 = toybox.matrix_exp_by_diagonalization(a)
        e_a3 = toybox.matrix_exp_by_symmetric_diagonalization(a)
        self.assert_(comp.norm_frobenius(e_a1-e_a2)
                     / comp.norm_frobenius(e_a1)
                     / comp.norm_frobenius(e_a2) <= 1e-15)
        self.assert_(comp.norm_frobenius(e_a1-e_a3)
                     / comp.norm_frobenius(e_a1)
                     / comp.norm_frobenius(e_a3) <= 1e-15)

    def test_matrix_exp(self):
        self.for_all_dtypes(self.do_test_matrix_exp)

    def do_test_eigenvectors_hermitian(self, dtype):
        size = 100

        a = make_random_spd_matrix(size, dtype)
        q, w = comp.diagonalize_hermitian(a)
        w2 = comp.eigenvalues_hermitian(a)

        self.assert_(abs(sum(w) - sum(w2)) < 1e-12)

        d = num.zeros(a.shape, a.dtype)
        for i in range(size):
            d[i,i] = w[i]
        self.assert_small(a - q*d*q.H)

    def test_eigenvectors_hermitian(self):
        self.for_all_dtypes(self.do_test_eigenvectors_hermitian)

    def do_test_eigenvectors(self, dtype):
        size = 100

        #a = make_random_full_matrix(size, dtype)
        a = tmd.eigvmat[dtype]

        evecs, evals = comp.diagonalize(a)
        evals2 = comp.eigenvalues(a)

        self.assert_(abs(sum(evals) - sum(evals2)) < 1e-12)

        d = num.zeros(a.shape, num.Complex)
        for i in range(size):
            d[i,i] = evals[i]
        self.assert_small(a*evecs - evecs* d)

    def test_eigenvectors(self):
        self.for_all_dtypes(self.do_test_eigenvectors)

    def do_test_bicgstab(self, dtype):
        # real case fails sometimes
        size = 30

        #A = make_random_full_matrix(size, dtype)
        #b = make_random_vector(size, dtype)
        #print repr(A)
        #print repr(b)

        # bicgstab is prone to failing on bad matrices
        A = tmd.bicgmat[dtype]
        b = tmd.bicgvec[dtype]

        A_op = op.MatrixOperator.make(A)
        bicgstab_op = op.BiCGSTABOperator.make(A_op, 40000, 1e-10)
        #bicgstab_op.debug_level = 1
        x = num.zeros((size,), dtype)

        initial_resid = comp.norm_2(b - A*x)
        bicgstab_op.apply(b, x)
        end_resid = comp.norm_2(b - A*x)
        self.assert_(end_resid/initial_resid < 1e-10)

    def test_bicgstab(self):
        self.for_all_dtypes(self.do_test_bicgstab)

    def test_complex_adaptor(self):
        size = 40

        a = make_random_full_matrix(size, num.Complex)
        a_op = op.MatrixOperator.make(a)
        a2_op = toybox.adapt_real_to_complex_operator(
            op.MatrixOperator.make(a.real), 
            op.MatrixOperator.make(a.imaginary))

        for i in range(20):
            b = make_random_vector(size, num.Complex)
            result1 = num.zeros((size,), num.Complex)
            result2 = num.zeros((size,), num.Complex)
            a_op.apply(b, result1)
            a2_op.apply(b, result2)
            self.assert_(comp.norm_2(result1 - result2) < 1e-11)

    def do_test_interpolation(self, dtype):
        size = 4

        def eval_at(x):
            # Horner evaluation
            result = 0.
            for i in coefficients[::-1]:
                result = i + x*result
            return result

        # generate random polynomial and random points

        #abscissae = make_random_vector(size, dtype)
        #coefficients = make_random_vector(size, dtype)
        #print repr(abscissae)
        #print repr(coefficients)
        abscissae = tmd.interpabs[dtype]
        coefficients = tmd.interpcoeff[dtype]

        # evaluate polynomial at abscissae
        values = num.array([eval_at(abscissa) for abscissa in abscissae])

        for to_x in tmd.interpx[dtype]:
            i_coeff = toybox.find_interpolation_coefficients(abscissae, to_x)
            f_x1 = eval_at(to_x)

            self.assert_(abs(i_coeff*values-f_x1) < 1e-7)

    def test_interpolation(self):
        self.for_all_dtypes(self.do_test_interpolation)

    def test_python_operator(self):
        class MyOperator(op.Operator(num.Float64)):
            def __init__(self, mat):
                op.Operator(num.Float64).__init__(self)
                self.Matrix = mat

            def size1(self):
                return self.Matrix.shape[0]

            def size2(self):
                return self.Matrix.shape[1]

            def apply(self, before, after):
                after[:] = self.Matrix * before

        size = 100

        A = make_random_spd_matrix(size, num.Float)
        Aop = MyOperator(A)
        b = make_random_vector(size, num.Float)
        cg_op = op.CGOperator.make(Aop, 4000, 1e-10)

        initial_resid = comp.norm_2(b)
        end_resid = comp.norm_2(b - A*cg_op(b))
        self.assert_(end_resid/initial_resid < 1e-10)

    def do_test_ssor(self, dtype):
        size = 200
        omega = 0.5
        a = make_random_spd_matrix(size, dtype)
        d = num.diagonal_matrix(a)
        l = d + omega * num.tril(a, -1)
        u = l.H
        d_inv = num.divide(1, num.diagonal(a))

        ssor_a = op.SSORPreconditioner.make(a, omega=omega)
        for i in range(5):
            vec = make_random_vector(size, dtype)
            ssor_vec = ssor_a(vec)
            vec_2 = 1/(omega*(2-omega))*(l*num.multiply(d_inv, (u*ssor_vec)))
            self.assert_small(vec - vec_2)

    def test_ssor(self):
        self.for_all_dtypes(self.do_test_ssor)
        
    def do_test_newton(self, dtype):
        def my_sin(x):
            if isinstance(x, complex):
                return cmath.sin(x)
            else:
                return math.sin(x)

        def my_cos(x):
            if isinstance(x, complex):
                return cmath.sin(x)
            else:
                return math.sin(x)

        def f(r):
            x = r[0]; y = r[1]
            # gradient of sin(x)*5*cos(y)+x**2+y**2
            return num.array([
                5*my_cos(x)*my_cos(y)+2*x, 
                -5*my_sin(x)*my_sin(y)+2*y])
        def fprime(r):
            x = r[0]; y = r[1]
            return num.array([
                [-5*my_cos(y)*my_sin(x)+2,
                    -5*my_cos(x)*my_sin(y)],
                [-5*my_cos(x)*my_sin(y),
                    -5*my_cos(y)*my_sin(x)+2]])
        result = toybox.find_vector_zero_by_newton(f, fprime, 
                num.array([1,2]))
        self.assert_small(f(result))

    def test_newton(self):
        self.for_all_dtypes(self.do_test_newton)
        
    def test_daskr(self):
        def f(t, y):
            return num.array([y[1], -y[0]])

        t, y, yp = toybox.integrate_ode(num.array([0,1]), f, 0, 10)
        times = num.array(t)
        analytic_solution = num.sin(times)
        y = num.array(y)[:,0]

        self.assert_(comp.norm_2(y-analytic_solution) < 1e-5)

    def test_assign_size_errors(self):
        u = num.zeros((10,10), num.Float)

        try:
            u[1] = num.ones((9,), num.Float)
            self.assert_(False)
        except ValueError:
            pass

        try:
            u[1:3] = num.ones((9,), num.Float)
            self.assert_(False)
        except ValueError:
            pass

        try:
            u[1:3] = num.ones((9,), num.Float)
            self.assert_(False)
        except ValueError:
            pass

        try:
            u[1:2, 1:2] = num.ones((1,2), num.Float)
            self.assert_(False)
        except ValueError:
            pass

        try:
            u[1:2, 1:2] = num.ones((2,1), num.Float)
            self.assert_(False)
        except ValueError:
            pass

        v = num.zeros((10,10), num.Float)

        try:
            v[1:2] = num.ones((2,), num.Float)
            self.assert_(False)
        except ValueError:
            pass

        try:
            v[1:2] = num.ones((2,1), num.Float)
            self.assert_(False)
        except ValueError:
            pass

    def do_test_symmetric_fun_calculus(self, dtype):
        size = 10
        a = make_random_spd_matrix(size, dtype)

        s = toybox.apply_f_to_symmetric(math.sqrt, a)
        self.assert_small(s*s-a)

    def test_symmetric_fun_calculus(self):
        self.for_all_dtypes(self.do_test_symmetric_fun_calculus)

    def test_permutation_matrix(self):
        from random import shuffle

        n = 37

        permut = range(37)
        shuffle(permut)

        fp = num.permutation_matrix(from_indices=permut)
        tp = num.permutation_matrix(to_indices=permut)
        
        self.assert_(fp.T == tp)

        e = [num.unit_vector(n, i) for i in range(n)]

        for i in range(n):
            self.assert_small(tp*e[i]-e[permut[i]])
            self.assert_small(fp*e[permut[i]]-e[i])
            
if __name__ == '__main__':
    unittest.main()
