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




# some tools ------------------------------------------------------------------
def delta(x,y):
    if x == y:
        return 1
    else:
        return 0




def sp(x,y):
    return num.innerproduct(num.conjugate(y),x)




def norm2squared(x):
    val = num.innerproduct(num.conjugate(x),x)
    try:
      return val.real
    except AttributeError:
      # whoops. not complex
      return val





def norm2(x):
    scalar_product = num.innerproduct(num.conjugate(x),x)
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
