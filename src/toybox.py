#
#  Copyright (c) 2004-2006
#  Andreas Kloeckner
#
#  Permission to use, copy, modify, distribute and sell this software
#  and its documentation for any purpose is hereby granted without fee,
#  provided that the above copyright notice appear in all copies and
#  that both that copyright notice and this permission notice appear
#  in supporting documentation.  The authors make no representations
#  about the suitability of this software for any purpose.
#  It is provided "as is" without express or implied warranty.
#




"""
PyLinear's module with experimental functionality.

There are many valuable capabilities in here, but they are either
still experimental or just not an official part of PyLinear yet.
"""




import pylinear
import pylinear.array as num
import pylinear.operator as op
import pylinear.computation as comp
import pylinear._operation as _op

import math
import pylinear.linear_algebra as la
import pylinear.iteration as iteration




def adapt_real_to_complex_operator(real_part, imaginary_part):
    if real_part.dtype != imaginary_part.dtype:
        raise TypeError, "outer and inner must have matching dtypes"
    return _op.ComplexMatrixOperatorAdaptorFloat64(real_part, imaginary_part)




# polynomial fits -------------------------------------------------------------
def vandermonde(vector, degree = None):
    if degree is None:
        degree = len(vector)

    mat = num.zeros((len(vector), degree+1), vector.dtype)
    for i, v in enumerate(vector):
        for power in range(degree+1):
            mat[i,power] = v**power
    return mat




def fit_polynomial(x_vector, data_vector, degree):
    vdm = vandermonde(x_vector, degree)
    vdm2 = vdm.H * vdm
    rhs = vdm.H * data_vector
    return vdm2 <<num.solve>> rhs




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




# interpolation ---------------------------------------------------------------
def find_interpolation_coefficients(x_vector, to_x):
    """ This routine finds the coefficients with which you
    need to dot the values of a function at the points
    given in x_vector to find the value at to_x, using
    polynomial interpolation.
    """
    vm = vandermonde(x_vector, degree = len(x_vector) - 1)
    vm_x = vandermonde(num.array([to_x], x_vector.dtype), len(x_vector)-1)[0]
    return vm.T <<num.solve>> vm_x

    
    
    
# Jacobi rotation -------------------------------------------------------------
def _conjugate(value):
    try:
        return value.conjugate()
    except AttributeError:
        return value


  

def _realPart(value):
    try:
        return value.real
    except AttributeError:
        return value



def _imaginaryPart(value):
    try:
        return value.imag
    except AttributeError:
        return 0.


  

class RotationShapeMatrix(object):
    def __init__(self, i, j, ii, ij, ji, jj):
        self.I = i
        self.J = j
        self.II = ii
        self.IJ = ij
        self.JI = ji
        self.JJ = jj

    def _hermite(self):
        return RotationShapeMatrix(self.I, self.J,
                _conjugate(self.II), _conjugate(self.JI),
                _conjugate(self.IJ), _conjugate(self.JJ))

    H = property(_hermite)

    def applyFromLeft(self, mat):
        row_i = mat[self.I]
        row_j = mat[self.J]

        mat[self.I] = row_i * self.II + row_j * self.IJ
        mat[self.J] = row_j * self.JJ + row_i * self.JI

    def applyFromRight(self, mat):
        col_i = mat[:,self.I]
        col_j = mat[:,self.J]

        mat[:,self.I] = col_i * self.II + col_j * self.JI
        mat[:,self.J] = col_j * self.JJ + col_i * self.IJ




def makeJacobiRotation(i, j, cos, sin):
    return RotationShapeMatrix(i,j,
            cos, _conjugate(sin),
            -sin, _conjugate(cos))
        



def _sign(x):
    return x / abs(x)

def diagonalize_jacobi(matrix, compute_vectors = True, 
                       observer = iteration.make_observer(rel_goal = 1e-10)):
    """This executes a simple sequence of Jacobi transform on the 
    given symmetric positive definite matrix. It returns a tuple 
    of matrices Q, D. Q is a unitary matrix that contains the
    eigenvectors of the given matrix in its columns.

    The following invariant holds:
    matrix = Q * D * Q^H.
    """

    # Simple Jacobi

    rows, columns = matrix.shape
    tc = matrix.dtype

    def off_diag_norm_squared(a):
        result = 0
        for i,j in a.indices():
            if i != j:
                result += abs(a[i,j])**2
        return result

    if compute_vectors:
        q = num.identity(rows, tc)
    else:
        q = None
    mymatrix = matrix.copy()

    if rows == 1 and columns == 1:
        return  q, mymatrix

    observer.reset()
    try:
        while True:
            observer.add_data_point(math.sqrt(off_diag_norm_squared(mymatrix)))
            for i in range(rows):
                for j in range(0,min(columns,i)):
                    if abs(mymatrix[i,j]) < 1e-10:
                        continue

                    theta = (mymatrix[j,j] - mymatrix[i,i])/mymatrix[i,j]
                    t = _sign(theta)/(abs(theta)+math.sqrt(abs(theta)**2+1))

                    other_cos = 1 / math.sqrt(abs(t)**2 + 1)
                    other_sin = t * other_cos

                    rot = makeJacobiRotation(i, j, other_cos, other_sin)
                    rot.H.applyFromLeft(mymatrix)
                    rot.applyFromRight(mymatrix)

                    if compute_vectors:
                        rot.applyFromRight(q)
    except iteration.IterationSuccessful:
        return q, mymatrix




def codiagonalize(matrices, observer = iteration.make_observer(stall_thresh = 1e-5, 
                                                               rel_goal = 1e-10)):
    """This executes the generalized Jacobi process from the research
    papers quoted below. It returns a tuple Q, diagonal_matrices, 
    achieved_tolerance.

    Q is a unitary matrix that contains the eigenvectors of the given 
    matrices in its columns, provided that complete diagonalization 
    could be achieved. diagonal_matrices contains the elements of
    `matrices', diagonalized as far as possible. achieved_tolerance,
    lastly, represents the amount of progress made. If this value is
    less than the `tolerance' parameter, convergence was achieved.
    Otherwise, it was decided that progress was no longer being made
    and the iteration was aborted or the max_iterations number was hit.

    The following invariant holds for each element `matrix' in matrices
    and `D' in diagonal_matrices.
    and : matrix = Q * D * Q^H.

    For one matrix, this reduces to the diagonalize() function in this
    module. 

    The matrices list is not modified, nor are any of its constituent
    matrices.

    Algorithm from:

    A. Bunse-Gerstner, R. Byers, V. Mehrmann:
    Numerical methods for simultaneous diagonalization
    SIAM J. of Matrix Analysis and Applications, Vol. 14, No. 4, 927-949
    (1993)

    J.-F. Cardoso, A. Souloumiac, Jacobi Angles for Simultaneous
    Diagonalization, also published by SIAM.
    """

    rows, columns = matrices[0].shape
    tc = matrices[0].dtype
    for mat in matrices[1:]:
        assert mat.shape == (rows, columns)
        assert mat.dtype == tc

    def off_diag_norm_squared(a):
        result = 0
        for i,j in a.indices():
            if i != j:
                result += abs(a[i,j])**2
        return result

    q = num.identity(rows, tc)
    norm_before = sum([off_diag_norm_squared(mat) for mat in matrices])
    mymats = [mat.copy() for mat in matrices]
    residual = math.sqrt(sum([off_diag_norm_squared(mat) for mat in mymats]))

    observer.reset()

    try:
        while True:
            observer.add_data_point(residual)
            for i in range(rows):
                for j in range(0,min(columns,i)):
                    g = num.zeros((3,3), num.Float)
                    for a in mymats:
                        h = num.array([_realPart(a[i,i] - a[j,j]),
                                       _realPart(a[i,j] + a[j,i]),
                                       -_imaginaryPart(a[j,i] - a[i,j])])
                        g += num.outerproduct(h, h)

                    u, diag_vec = la.Heigenvectors(g)

                    max_index = None
                    for index in range(3):
                        curval = abs(diag_vec[index])
                        if max_index is None or curval > current_max:
                            max_index = index
                            current_max = curval

                    if max_index is None:
                        continue

                    # eigenvector belonging to largest eigenvalue
                    bev = u[:,max_index]

                    if bev[0] < 0:
                        bev = - bev

                    r = comp.norm_2(bev)
                    if (bev[0] + r)/r < 1e-7:
                        continue

                    cos = math.sqrt((bev[0]+r)/(2*r))
                    sin = bev[1] 
                    if tc is not num.Float:
                        sin -= 1j*bev[2]
                    sin /= math.sqrt(2*r*(bev[0]+r))

                    rot = makeJacobiRotation(i, j, cos, sin)
                    rot_h = rot.H
                    for mat in mymats:
                        rot.applyFromLeft(mat)
                        rot_h.applyFromRight(mat)

                    rot_h.applyFromRight(q)

            residual = math.sqrt(sum([off_diag_norm_squared(mat) for mat in mymats]))

    except iteration.IterationStalled:
        return q, mymats, math.sqrt(residual/norm_before)
    except iteration.IterationSuccessful:
        return q, mymats, math.sqrt(residual/norm_before)





# functional calculus for symmetric matrices ----------------------------------
def apply_f_to_symmetric(f, a):
    # a has to be symmetric
    h,w = a.shape
    assert h == w

    q, w = la.Heigenvectors(a)
    new_w = num.zeros_like(w)
    for i, v in enumerate(w):
        new_w[i] = f(v)
    return q*num.diagonal_matrix(new_w)*q.H
    
        
    

# matrix exponentials ---------------------------------------------------------
def matrix_exp_by_series(a, eps = 1e-15):
    h,w = a.shape
    assert h == w

    a_frob = comp.norm_frobenius(a)
    
    last_result = num.identity(h, a.dtype)
    result = last_result.copy()

    current_power_of_a = a

    factorial = 1
    n = 1

    while True:
        result += current_power_of_a * (1./factorial)

        if comp.norm_frobenius(result - last_result)/a_frob < eps:
            return result

        n += 1
        last_result = result.copy()
        factorial *= n
        current_power_of_a = current_power_of_a * a
    
        
    

def matrix_exp_by_symmetric_diagonalization(a):
    # a has to be symmetric
    h,w = a.shape
    assert h == w

    q, w = la.Heigenvectors(a)
    return q*num.diagonal_matrix(num.exp(w))*q.H
    
        
    

def matrix_exp_by_diagonalization(a):
    h,w = a.shape
    assert h == w

    v, w = la.eigenvectors(a)
    v_t = v.T
    return (v_t <<num.solve>> (num.diagonal_matrix(num.exp(w)) * v_t)).T




# matrix type tests -----------------------------------------------------------
def hermiticity_error(mat):
    return comp.norm_frobenius(mat.H - mat)

def skewhermiticity_error(mat):
    return comp.norm_frobenius(mat.H + mat)

def symmetricity_error(mat):
    return comp.norm_frobenius(mat.T - mat)

def skewsymmetricity_error(mat):
    return comp.norm_frobenius(mat.T + mat)

def unitariety_error(mat):
    return identity_error(mat.H * mat)

def orthogonality_error(mat):
    return identity_error(mat.T * mat)

def identity_error(mat):
    id_mat = num.identity(mat.shape[0], mat.dtype)
    return comp.norm_frobenius(mat - id_mat)





# Numerical algorithms -------------------------------------------------------
def find_zero_by_newton(f, fprime, x_start, tolerance = 1e-12, maxit = 10):
    it = 0
    while it < maxit:
        it += 1
        f_value = f(x_start)
        if math.fabs(f_value) < tolerance:
            return x_start
        x_start -= f_value / fprime(x_start)
    raise RuntimeError, "Newton iteration failed, a zero was not found"




def find_vector_zero_by_newton(f, fprime, x_start, tolerance = 1e-12, maxit = 100):
    it = 0
    while it < maxit:
        it += 1
        f_value = f(x_start)
        if comp.norm_2(f_value) < tolerance:
            return x_start
        x_start -= fprime(x_start) <<num.solve>> f_value
    raise RuntimeError, "Newton iteration failed, a zero was not found"




def distance_to_line(start_point, direction, point):
    # Ansatz: start_point + alpha * direction 
    # <start_point + alpha * direction - point, direction> = 0!
    alpha = - num.innerproduct(start_point - point, direction) / \
            comp.norm_2_squared(direction)
    foot_point = start_point + alpha * direction
    return comp.norm_2(point - foot_point), alpha




def angle_cosine_between_vectors(vec1, vec2):
    return vec1*vec2.H / (comp.norm_2(vec1)*comp.norm_2(vec2))




def interpolate_vector_list(vectors, inbetween_points):
    if len(vectors) == 0:
        return []

    result = [vectors[0]]
    last_vector = vectors[0]
    for vector in vectors[1:]:
        for i in range(inbetween_points):
            result.append(last_vector + (vector-last_vector) \
                          * float(i+1) \
                          / float(inbetween_points+1))
        result.append(vector)
        last_vector = vector
    return result




def make_rotation_matrix(radians, n = 2, axis1 = 0, axis2 = 1, dtype = num.Float):
    mat = num.identity(n, dtype)
    mat[axis1,axis1] = math.cos(radians)
    mat[axis2,axis1] = math.sin(radians)
    mat[axis1,axis2] = -math.sin(radians)
    mat[axis2,axis2] = math.cos(radians)
    return mat




def get_parallelogram_volume(vectors):
    if vectors[0].shape[0] == 2:
        return vectors[0][0] * vectors[1][1] - vectors[1][0] * vectors[0][1]
    else:
        raise RuntimeError, "not implemented"




def unit_vector(i, dim, dtype = num.Float):
    uvec = num.zeros((dim,), dtype)
    uvec[i] = 1
    return uvec




def conjugate(value):
    try:
        return value.conjugate()
    except AttributeError:
        return value




# ODEs -----------------------------------------------------------------------
def runge_kutta_step(start, dt, f):
    w1 = start + dt/2 * f(start)
    w2 = start + dt/2 * f(w1)
    w3 = start + dt * f(w2)
    return 1./3 * (-start+w1+2*w2+w3+dt/2*f(w3))




if pylinear.has_daskr():
    DAE = _op.DAE
    DAESolver = _op.DAESolver

    def integrate_dae(dae, t, y0, yprime0, t_end, steps=100, 
            intermediate_steps=False):
        solver = DAESolver(dae)

        times = [t]
        y_data = [y0]
        yprime_data = [yprime0]

        dt = float(t_end-t)/steps

        if intermediate_steps:
            solver.want_intermediate_steps = True

        t_start = t

        y = y0.copy()
        yprime = yprime0.copy()
        while t < t_end:
            progress_in_current_timestep = (t-t_start)%dt
            if progress_in_current_timestep > 0.99 * dt:
                next_timestep = t+2*dt-progress_in_current_timestep
            else:
                next_timestep = t+dt-progress_in_current_timestep

            state, t = solver.step(t, next_timestep, y, yprime0)

            times.append(t)
            y_data.append(y.copy())
            yprime_data.append(yprime.copy())

        return times, y_data, yprime_data




    def integrate_ode(initial, f, t, t_end, steps=100):
        n = len(f(t, initial))

        class my_dae(DAE):
            def dimension(self):
                return n

            def residual(self, t, y, yprime):
                #print t, y
                return yprime - f(t, y)

        return integrate_dae(my_dae(), t, initial, f(t, initial), t_end, steps=steps)


# Obscure stuff --------------------------------------------------------------
def write_matrix_as_csv(filename, matrix):
    mat_file = file(filename, "w")
    h,w = matrix.shape
    for row in range(0, h):
        for column in range(0, w):
            mat_file.write("%f," % matrix[ row, column ])
    mat_file.write("\n")




def shift(vec, dist):
    result = vec.copy()
    N = len(vec)
    dist = dist % N
    if dist > 0:
        result[dist:] = vec[:N-dist]
        result[:dist] = vec[N-dist:]
    return result


        



