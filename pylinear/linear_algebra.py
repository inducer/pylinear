import pylinear.matrices as num
import pylinear.algorithms as algo




def solve_linear_equations_umf(mat, rhs):
    typecode = mat.typecode()
    h,w = mat.shape
    umf_operator = algo.makeUMFPACKMatrixOperator(
        num.asarray(mat, mat.typecode(), num.SparseExecuteMatrix))

    temp = num.zeros((h,), typecode)
    if len(rhs.shape) == 1:
        umf_operator.apply(rhs, temp)
        return temp
    else:
        rhh, rhw = rhs.shape
        
        solution = num.zeros(rhs.shape, typecode)
        assert rhh == h
        for col in range(rhw):
            umf_operator.apply(rhs[:,col], temp)
            solution[:,col] = temp
        return solution




def solve_linear_equations(mat, rhs):
    typecode = mat.typecode()
    h,w = mat.shape
    l, u, permut, sign = algo.lu(mat)

    temp = num.zeros((h,), typecode)
    if len(rhs.shape) == 1:
        for i in range(h):
            temp[i] = rhs[permut[i]]
        return u.solveUpper(l.solveLower(temp))
    else:
        rhh, rhw = rhs.shape
        
        solution = num.zeros(rhs.shape, typecode)
        assert rhh == h
        for col in range(rhw):
            for i in range(h):
                temp[i] = rhs[permut[i],col]
                solution[:,col] = u.solveUpper(l.solveLower(temp))
        return solution




def inverse(mat):
    w,h = mat.shape
    assert h == w
    return solve_linear_equations(mat, num.identity(h, mat.typecode()))




def determinant(mat):
    h,w = mat.shape
    assert h == w
    if h == 2:
        return mat[0,0]*mat[1,1] - mat[1,0]*mat[0,1]
    else:
        try:
            l,u, permut, sign = algo.lu(mat)
            
            product = 1
            for i in range(h):
                product *= u[i,i]

            return product * sign
        except RuntimeError:
            # responds to the "is singular" exception
            return 0



singular_value_decomposition = algo.singular_value_decomposition




def Heigenvalues(mat, upper = True):
    q, w = algo.Heigenvectors(False, upper, mat)
    return w

def Heigenvectors(mat, upper = True):
    return algo.Heigenvectors(True, upper, mat)

def eigenvalues(mat):
    w, vl, vr = algo.eigenvectors(False, False, mat)
    return w

def eigenvectors(mat):
    w, vl, vr = algo.eigenvectors(False, True, mat)
    return vr, w





def cholesky_decomposition(a):
    return algo.cholesky(a)
