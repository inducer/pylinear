import sys
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools


def off_diag_norm_squared(a):
    result = 0
    for i,j in a.indices():
        if i != j:
            result += abs(a[i,j])**2
    return result

def off_diag_norms_squared(matrices):
    return sum([off_diag_norm_squared(mat) for mat in matrices])

size = 20
tc = num.Float

matrices = [mtools.makeFullRandomMatrix(size, tc) for i in range(2)]

print off_diag_norms_squared(matrices)
q, mats, achieved = mtools.codiagonalize(matrices)
print off_diag_norms_squared(mats)
