import math, cmath, random, types
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.algorithms as algo
import pylinear.iteration as iteration




# polynomial fits -------------------------------------------------------------
def vandermonde(vector, degree = None):
    size, = vector.shape

    if degree is None:
        degree = size

    mat = num.zeros((size, degree+1), vector.typecode())
    for i,v in zip(range(size), vector):
        for power in range(degree+1):
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


  

class tRotationShapeMatrix:
    def __init__(self, i, j, ii, ij, ji, jj):
        self.I = i
        self.J = j
        self.II = ii
        self.IJ = ij
        self.JI = ji
        self.JJ = jj

    def hermite(self):
        return tRotationShapeMatrix(self.I, self.J,
                                    _conjugate(self.II), _conjugate(self.JI),
                                    _conjugate(self.IJ), _conjugate(self.JJ))

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
    return tRotationShapeMatrix(i,j,
                                cos, _conjugate(sin),
                                -sin, _conjugate(cos))
        



def printMatrixInGrid(a):
    h,w = a.shape
    for i in range(h):
        for j in range(w):
            print "%.6f  " % a[i,j],
        print
def printComplexMatrixInGrid(a):
    h,w = a.shape
    for i in range(h):
        for j in range(w):
            print "%.3f %.3fj  " % (a[i,j].real, a[i,j].imag),
        print

def _sign(x):
    return x / abs(x)

def diagonalize(matrix, compute_vectors = True, observer = iteration.makeObserver(rel_goal = 1e-10)):
    """This executes a simple sequence of Jacobi transform on the 
    given symmetric positive definite matrix. It returns a tuple 
    of matrices Q, D. Q is a unitary matrix that contains the
    eigenvectors of the given matrix in its columns.

    The following invariant holds:
    matrix = Q * D * Q^H.
    """

    # Simple Jacobi

    rows, columns = matrix.shape
    tc = matrix.typecode()

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
            observer.addDataPoint(math.sqrt(off_diag_norm_squared(mymatrix)))
            for i in range(rows):
                for j in range(0,min(columns,i)):
                    if abs(mymatrix[i,j]) < 1e-10:
                        continue

                    theta = (mymatrix[j,j] - mymatrix[i,i])/mymatrix[i,j]
                    t = _sign(theta)/(abs(theta)+math.sqrt(abs(theta)**2+1))

                    other_cos = 1 / math.sqrt(abs(t)**2 + 1)
                    other_sin = t * other_cos

                    rot = makeJacobiRotation(i, j, other_cos, other_sin)
                    rot.hermite().applyFromLeft(mymatrix)
                    rot.applyFromRight(mymatrix)

                    if compute_vectors:
                        rot.applyFromRight(q)
    except iteration.tIterationSuccessful:
        return q, mymatrix




def codiagonalize(matrices, observer = iteration.makeObserver(stall_thresh = 1e-5, rel_goal = 1e-10)):
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

    mm = num.matrixmultiply

    rows, columns = matrices[0].shape
    tc = matrices[0].typecode()
    for mat in matrices[1:]:
        assert mat.shape == (rows, columns)
        assert mat.typecode() == tc

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
            observer.addDataPoint(residual)
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

                    r = norm2(bev)
                    if (bev[0] + r)/r < 1e-7:
                        continue

                    cos = math.sqrt((bev[0]+r)/(2*r))
                    sin = bev[1] 
                    if tc is not num.Float:
                        sin -= 1j*bev[2]
                    sin /= math.sqrt(2*r*(bev[0]+r))

                    rot = makeJacobiRotation(i, j, cos, sin)
                    rot_h = rot.hermite()
                    for mat in mymats:
                        rot.applyFromLeft(mat)
                        rot.hermite().applyFromRight(mat)

                    rot_h.applyFromRight(q)

            residual = math.sqrt(sum([off_diag_norm_squared(mat) for mat in mymats]))

    except iteration.tIterationStalled:
        return q, mymats, math.sqrt(residual/norm_before)
    except iteration.tIterationSuccessful:
        return q, mymats, math.sqrt(residual/norm_before)





# some tools ------------------------------------------------------------------
def solve_linear_system_cg(matrix, vector):
    m_inv = algo.makeCGMatrixOperator(algo.makeMatrixOperator(matrix), 
                                      matrix.shape[0]*2)
    return algo.applyMatrixOperator(m_inv, vector)
    



def entrySum(a):
    result = 0
    for i,j in a.indices():
        result += a[i,j]
    return result




def frobeniusNormSquared(a):
    result = 0
    for i,j in a.indices():
        result += abs(a[i,j])**2
    return result




def frobeniusNorm(a):
    return math.sqrt(frobeniusNormSquared(a))




def matrixExp(a, eps = 1e-15):
    h,w = a.shape
    assert h == w

    a_frob = frobeniusNorm(a)
    
    last_result = num.identity(h, a.typecode())
    result = last_result.copy()

    current_power_of_a = a

    factorial = 1
    n = 1

    while True:
        result += current_power_of_a * (1./factorial)

        if frobeniusNorm(result - last_result)/a_frob < eps:
            return result

        n += 1
        last_result = result.copy()
        factorial *= n
        current_power_of_a = num.matrixmultiply(current_power_of_a, a)
    
        
    

def matrixExpBySymmetricDiagonalization(a):
    # a has to be symmetric
    h,w = a.shape
    assert h == w

    q, w = la.Heigenvectors(a)
    d = num.zeros(a.shape, a.typecode())
    for i in range(h):
        d[i,i] = cmath.exp(w[i])
    mm = num.matrixmultiply
    return mm(q, mm(d, hermite(q)))
    
        
    

def matrixExpByDiagonalization(a):
    h,w = a.shape
    assert h == w

    v, w = la.eigenvectors(a)
    e_d = num.zeros(a.shape, a.typecode())
    v_t = num.transpose(v)
    for i in range(h):
        e_d[i,i] = cmath.exp(w[i])
    mm = num.matrixmultiply
    return num.transpose(la.solve_linear_equations(v_t, mm(e_d, v_t)))




def delta(x,y):
    if x == y:
        return 1
    else:
        return 0




def sp(x,y):
    return num.innerproduct(x, num.conjugate(y))




def norm2squared(x):
    val = num.innerproduct(x,num.conjugate(x))
    try:
      return val.real
    except AttributeError:
      # whoops. not complex
      return val





def norm2(x):
    scalar_product = num.innerproduct(x,num.conjugate(x))
    try:
        return math.sqrt(scalar_product.real)
    except AttributeError:
        # whoops. not complex
        return math.sqrt(scalar_product)




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


    

# random matrices -------------------------------------------------------------
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
    vectors = orthogonalize(vectors)

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


  

def makeRandomSkewHermitianMatrix(size, typecode):
    a = num.zeros((size, size), typecode)
    # fill diagonal
    if typecode is num.Complex:
        for i in range(size):
            a[i,i] = 1j*random.normalvariate(0,10)

    # fill rest
    for i in range(size):
        for j in range(i):
            value = random.normalvariate(0,10)
            if typecode is num.Complex:
                value += 1j*random.normalvariate(0,10)
            a[i,j] = value
            a[j,i] = -_conjugate(value)
    return a




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




# matrix type tests -----------------------------------------------------------
def hermiticityError(mat):
    return frobeniusNorm(num.hermite(mat) - mat)

def skewHermiticityError(mat):
    return frobeniusNorm(num.hermite(mat) + mat)

def symmetricityError(mat):
    return frobeniusNorm(num.transpose(mat) - mat)

def skewSymmetricityError(mat):
    return frobeniusNorm(num.transpose(mat) + mat)

def unitarietyError(mat):
    mm = num.matrixmultiply
    return identityError(mm(num.hermite(mat), mat))

def orthogonalityError(mat):
    mm = num.matrixmultiply
    return identityError(mm(num.transpose(mat), mat))

def identityError(mat):
    id = num.identity(mat.shape[0], mat.typecode())
    return frobeniusNorm(mat - id)

