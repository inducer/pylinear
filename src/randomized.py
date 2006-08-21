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
PyLinear's module for random matrices.
"""




import random
import pylinear.array as num
import pylinear.computation as comp




def write_random_vector(vec):
    size, = vec.shape
    for i in range(size):
        value = random.normalvariate(0,10)
        if vec.dtype == num.Complex64:
            value += 1j*random.normalvariate(0,10)
        vec[i] = value




def make_random_vector(size, dtype):
    vec = num.zeros((size,), dtype)
    write_random_vector(vec)
    return vec




def make_random_onb(size, dtype):
    vectors = [ makeRandomVector(size, dtype) for i in range(size) ]
    vectors = comp.orthogonalize(vectors)

    for i in range(size):
        for j in range(size):
            assert abs(delta(i,j) - sp(vectors[i], vectors[j])) < 1e-12
    return vectors





def make_random_orthogonal_matrix(size, dtype):
    vectors = []
    for i in range(size):
        v = num.zeros((size,), dtype)
        write_random_vector(v)
        vectors.append(v)

    orth_vectors = comp.orthogonalize(vectors)

    mat = num.zeros((size,size), dtype)
    for i in range(size):
        mat[:,i] = orth_vectors[i]

    return mat


  

def make_random_skewhermitian_matrix(size, dtype):
    a = num.zeros((size, size), dtype)
    # fill diagonal
    if dtype is num.Complex:
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
            if dtype is num.Complex:
                value += 1j*random.normalvariate(0,10)
            a[i,j] = value
            a[j,i] = -_conjugate(value)
    return a




def make_random_spd_matrix(size, dtype):
    eigenvalues = make_random_vector(size, dtype)
    eigenmat = num.zeros((size,size), dtype)
    for i in range(size):
        eigenmat[i,i] = abs(eigenvalues[i])

    orthomat = make_random_orthogonal_matrix(size, dtype)
    return orthomat.H * eigenmat *orthomat




def make_random_full_matrix(size, dtype):
    result = num.zeros((size, size), dtype)
    
    for row in range(size):
        for col in range(size):
            value = random.normalvariate(0,10)
            if dtype == num.Complex64:
                value += 1j*random.normalvariate(0,10)

            result[row,col] = value
    return result




def make_random_matrix(size, dtype, flavor = num.DenseMatrix):
    result = num.zeros((size, size), dtype, flavor)
    elements = size ** 2 / 10

    for i in range(elements):
        row = random.randrange(0, size)
        col = random.randrange(0, size)
    
        value = random.normalvariate(0,10)
        if dtype == num.Complex64:
            value += 1j*random.normalvariate(0,10)

        result[row,col] += value
    return result

