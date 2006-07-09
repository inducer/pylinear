"""
PyLinear's module for random matrices.
"""




import random
import pylinear.array as num
import pylinear.computation as comp




def write_random_vector(vec):
    size, = vec.shape
    for i in range(size):
        value = random.normalvariate(0,10)
        if vec.typecode() == num.Complex64:
            value += 1j*random.normalvariate(0,10)
        vec[i] = value




def make_random_vector(size, typecode):
    vec = num.zeros((size,), typecode)
    write_random_vector(vec)
    return vec




def make_random_onb(size, typecode):
    vectors = [ makeRandomVector(size, typecode) for i in range(size) ]
    vectors = comp.orthogonalize(vectors)

    for i in range(size):
        for j in range(size):
            assert abs(delta(i,j) - sp(vectors[i], vectors[j])) < 1e-12
    return vectors





def make_random_orthogonal_matrix(size, typecode):
    vectors = []
    for i in range(size):
        v = num.zeros((size,), typecode)
        write_random_vector(v)
        vectors.append(v)

    orth_vectors = comp.orthogonalize(vectors)

    mat = num.zeros((size,size), typecode)
    for i in range(size):
        mat[:,i] = orth_vectors[i]

    return mat


  

def make_random_skewhermitian_matrix(size, typecode):
    a = num.zeros((size, size), typecode)
    # fill diagonal
    if typecode is num.Complex:
        for i in range(size):
            a[i,i] = 1j*random.normalvariate(0,10)

    def _conjugate(x):
        try:
            return x.conjugate()
        except AttributeError:
            return x

    # fill rest
    for i in range(size):
        for j in range(i):
            value = random.normalvariate(0,10)
            if typecode is num.Complex:
                value += 1j*random.normalvariate(0,10)
            a[i,j] = value
            a[j,i] = -_conjugate(value)
    return a




def make_random_spd_matrix(size, typecode):
    eigenvalues = make_random_vector(size, typecode)
    eigenmat = num.zeros((size,size), typecode)
    for i in range(size):
        eigenmat[i,i] = abs(eigenvalues[i])

    orthomat = make_random_orthogonal_matrix(size, typecode)
    return orthomat.H * eigenmat *orthomat




def make_random_full_matrix(size, typecode):
    result = num.zeros((size, size), typecode)
    
    for row in range(size):
        for col in range(size):
            value = random.normalvariate(0,10)
            if typecode == num.Complex64:
                value += 1j*random.normalvariate(0,10)

            result[row,col] = value
    return result




def make_random_matrix(size, typecode, flavor = num.DenseMatrix):
    result = num.zeros((size, size), typecode, flavor)
    elements = size ** 2 / 10

    for i in range(elements):
        row = random.randrange(0, size)
        col = random.randrange(0, size)
    
        value = random.normalvariate(0,10)
        if typecode == num.Complex64:
            value += 1j*random.normalvariate(0,10)

        result[row,col] += value
    return result

