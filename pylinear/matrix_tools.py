import math, random, types
import pylinear.matrices as num
import pylinear.linear_algebra as la




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

def diagonalize(matrix, tolerance = 1e-10, max_iterations = None):
    # Simple Jacobi

    rows, columns = matrix.shape
    tc = matrix.typecode()

    def off_diag_norm_squared(a):
        result = 0
        for i,j in a.indices():
            if i != j:
                result += abs(a[i,j])**2
        return result

    q = num.identity(rows, tc)
    norm_before = off_diag_norm_squared(matrix)
    mymatrix = matrix.copy()
    iterations = 0

    while off_diag_norm_squared(mymatrix) >= tolerance**2 * norm_before:
        for i in range(rows):
            for j in range(0,min(columns,i)):
                if abs(mymatrix[i,j]) < tolerance ** 3:
                    continue

                theta = (mymatrix[j,j] - mymatrix[i,i])/mymatrix[i,j]
                t = _sign(theta)/(abs(theta)+math.sqrt(abs(theta)**2+1))

                other_cos = 1 / math.sqrt(abs(t)**2 + 1)
                other_sin = t * other_cos

                rot = makeJacobiRotation(i, j, other_cos, other_sin)
                rot.applyFromRight(q)
                rot.hermite().applyFromLeft(mymatrix)
                rot.applyFromRight(mymatrix)

        iterations += 1
        if max_iterations and (iterations >= max_iterations):
            raise RuntimeError, "Jacobi diagonalization failed to converge"
    return q, mymatrix




def codiagonalize(matrices, tolerance = 1e-5, max_iterations = None):
    # From A. Bunse-Gerstner, R. Byers, V. Mehrmann:
    # Numerical methods for simultaneous diagonalization
    # SIAM J. of Matrix Analysis and Applications, Vol. 14, No. 4, 927-949
    # (1993)

    # For the determination of the Jacobi angles, see 
    # J.-F. Cardoso, A. Souloumiac, Jacobi Angles for Simultaneous
    # Diagonalization, also published by SIAM.

    rows, columns = matrices[0].shape
    tc = matrices[0].typecode()
    for mat in matrices[1:]:
        assert mat.shape == (height, width)
        assert mat.typecode() == tc

    def off_diag_norm_squared(a):
        result = 0
        for i,j in a.indices():
            if i != j:
                result += abs(a[i,j])**2
        return result

    q = num.identity(rows, num.Complex)
    frobsum = sum([frobeniusNormSquared(mat) for mat in matrices])
    mymats = [num.asarray(mat, num.Float).copy() for mat in matrices]
    iterations = 0

    print_all = False

    while sum([off_diag_norm_squared(mat) for mat in mymats]) \
          >= tolerance * frobsum:
        print "HA!", sum([off_diag_norm_squared(mat) for mat in mymats]) \

        for i in range(rows):
            for j in range(0,min(columns,i)):
                if True:
                    g = num.zeros((3,3), num.Complex)
                    for a in mymats:
                        h = num.array([a[i,i] - a[j,j],
                                       a[i,j] + a[j,i],
                                       1j * (a[j,i] - a[i,j])])
                        g += num.outerproduct(num.conjugate(h), h)
                    g = g.real
                    u, diag_mat = diagonalize(g)

                    max_index = None
                    for index in range(3):
                        if (not max_index or abs(diag_mat[index,index]) > current_max) \
                             and u[0,index] >= 0:
                            max_index = index
                            current_max = abs(diag_mat[index,index])

                    if max_index is None:
                        continue
                
                    # eigenvector belonging to largest eigenvalue
                    bev = u[:,max_index]
                    r = norm2(bev)
                    if (bev[0] + r)/r < 1e-7:
                        continue

                    cos = math.sqrt((bev[0]+r)/(2*r))
                    sin = (bev[1] - 1j*bev[2])/ math.sqrt(2*r*(bev[0]+r))

                a = mymats[0]
                theta = (a[j,j] - a[i,i])/a[i,j]
                t = _sign(theta)/(abs(theta)+math.sqrt(theta**2+1))

                other_cos = 1 / math.sqrt(t*t + 1)
                other_sin = t * other_cos

                print cos, "vs.", other_cos
                print sin, "vs.", other_sin

                rot = makeJacobiRotation(i, j, other_cos, other_sin)
                rot.applyFromRight(q)
                for mat in mymats:
                    rot.hermite().applyFromLeft(mat)
                    rot.applyFromRight(mat)

                if print_all:
                    print i,j
                    for mat in mymats:
                        printComplexMatrixInGrid(mat)
                    raw_input()

        sel = raw_input()
        print_all = False
        if sel == "p":
            for mat in mymats:
                printComplexMatrixInGrid(mat)
        if sel == "a":
            print_all = True

        iterations += 1
        if max_iterations and (iterations >= max_iterations):
            raise RuntimeError, "Codiagonalization failed to converge"
    return q, mymats





# some tools ------------------------------------------------------------------
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
    
        
    

def delta(x,y):
    if x == y:
        return 1
    else:
        return 0




def sp(x,y):
    return num.innerproduct(y,num.conjugate(x))




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
