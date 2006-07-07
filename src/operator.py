import pylinear
import pylinear.array as num
import pylinear.computation as comp
import pylinear._operation as _op

# operator parameterized types ------------------------------------------------
Operator = num.TypecodeParameterizedType(
  "MatrixOperator", _op.__dict__)
IdentityOperator = num.TypecodeParameterizedType(
  "IdentityMatrixOperator", _op.__dict__)
ScalarMultiplicationOperator = num.TypecodeParameterizedType(
  "ScalarMultiplicationMatrixOperator", _op.__dict__)

class _MatrixOperatorTypecodeFlavorParameterizedType:
    def is_a(self, instance):
        # FIXME
        raise NotImplementedError

    def __call__(self, typecode, flavor):
        # FIXME
        raise NotImplementedError

    def make(self, matrix):
        return _op.makeMatrixOperator(matrix)
MatrixOperator = _MatrixOperatorTypecodeFlavorParameterizedType()

class _CGTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.typecode(), w)
        if matrix_op.typecode() is not precon_op.typecode():
            raise TypeError, "matrix_op and precon_op must have matching typecodes"
        return self.TypeDict[matrix_op.typecode()](matrix_op, precon_op, max_it, tolerance)
    
CGOperator = _CGTypecodeParameterizedType("CGMatrixOperator", _op.__dict__)

class _BiCGSTABTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.typecode(), w)
        if matrix_op.typecode() is not precon_op.typecode():
            raise TypeError, "matrix_op and precon_op must have matching typecodes"
        return self.TypeDict[matrix_op.typecode()](matrix_op, precon_op, max_it, tolerance)
    
BiCGSTABOperator = _BiCGSTABTypecodeParameterizedType(
    "BiCGSTABMatrixOperator", _op.__dict__)

if pylinear.has_umfpack():
    class _UMFPACKTypecodeParameterizedType(num.TypecodeParameterizedType):
        def make(self, matrix):
            matrix.complete_index1_data()
            return self.TypeDict[matrix.typecode()](matrix)

    UMFPACKOperator = _UMFPACKTypecodeParameterizedType("UMFPACKMatrixOperator", 
                                                        _op.__dict__)

class _LUInverseOperator:
    def __init__(self, l, u, perm):
        assert l.shape[0] == l.shape[1]
        assert u.shape[0] == u.shape[1]
        assert l.shape[0] == u.shape[0]

        self.L = l
        self.U = u
        self.Permutation = perm

    def size1(self):
        return self.L.shape[0]
    
    def size2(self):
        return self.L.shape[1]

    def apply(self, before, after):
        temp = num.zeros((len(before),), before.typecode())
        for i in range(len(before)):
            temp[i] = before[self.Permutation[i]]
        after[:] = self.U.solve_upper(self.L.solve_lower(temp))

class _LUInverseOperatorFloat64(_LUInverseOperator, _op.MatrixOperatorFloat64):
    def __init__(self, l, u, perm):
        _LUInverseOperator.__init__(self, l, u, perm)
        _op.MatrixOperatorFloat64.__init__(self)

class _LUInverseOperatorComplex64(_LUInverseOperator, _op.MatrixOperatorComplex64):
    def __init__(self, l, u, perm):
        _LUInverseOperator.__init__(self, l, u, perm)
        _op.MatrixOperatorComplex64.__init__(self)

class _LUInverseTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, *args):
        if len(args) == 3:
            l, u, perm = args
        elif len(args) == 1:
            l, u, perm, sign = comp.lu(args[0])
        else:
            raise TypeError, "Invalid number of arguments"

        return self.TypeDict[l.typecode()](l, u, perm)

LUInverseOperator = _LUInverseTypecodeParameterizedType("_LUInverseOperator", 
                                                        globals())

class _SSORPreconditioner:
    def __init__(self, mat, omega=1):
        # mat needs to be symmetric
        assert mat.shape[0] == mat.shape[1]

        l = num.lower_left(mat)
        d = num.diagonal_matrix(mat)

        self.L = d + omega*l
        self.U = self.L.H
        self.DVector = num.diagonal(mat)
        self.Omega = omega

    def size1(self):
        return self.L.shape[0]
    
    def size2(self):
        return self.L.shape[1]

    def apply(self, before, after):
        after[:] = self.Omega * (2-self.Omega) * \
                   self.U.solve_upper(num.multiply(self.DVector, 
                                                  self.L.solve_lower(before)))

class _SSORPreconditionerFloat64(_SSORPreconditioner, 
                                 _op.MatrixOperatorFloat64):
    def __init__(self, *args, **kwargs):
        _SSORPreconditioner.__init__(self, *args, **kwargs)
        _op.MatrixOperatorFloat64.__init__(self)

class _SSORPreconditionerComplex64(_SSORPreconditioner, 
                                   _op.MatrixOperatorComplex64):
    def __init__(self, *args, **kwargs):
        _SSORPreconditioner.__init__(self, *args, **kwargs)
        _op.MatrixOperatorComplex64.__init__(self)

class _SSORPreconditionerTypecodeParameterizedType(num.TypecodeParameterizedType):
    def make(self, mat, *args, **kwargs):
        return num.TypecodeParameterizedType.make(
            self, mat.typecode(), mat, *args, **kwargs)

SSORPreconditioner = _SSORPreconditionerTypecodeParameterizedType(
    "_SSORPreconditioner", globals())


# operator operators ----------------------------------------------------------
_SumOfOperators = num.TypecodeParameterizedType(
  "SumOfMatrixOperators", _op.__dict__)
_ScalarMultiplicationOperator = num.TypecodeParameterizedType(
  "ScalarMultiplicationMatrixOperator", _op.__dict__)
_CompositeOfOperators = num.TypecodeParameterizedType(
  "CompositeMatrixOperator", _op.__dict__)




def _neg_operator(op):
    return _compose_operators(
        _ScalarMultiplicationOperator(op.typecode())(-1, op.shape[0]),
        op)

def _add_operators(op1, op2):
    return _SumOfOperators(op1.typecode())(op1, op2)

def _subtract_operators(op1, op2):
    return _add_operators(op1, _neg_operator(op2))

def _compose_operators(outer, inner):
    return _CompositeOfOperators(outer.typecode())(outer, inner)

def _multiply_operators(op1, op2):
    if num._is_number(op2):
        return _compose_operators(
            op1,
            _ScalarMultiplicationOperator(op1.typecode())(op2, op1.shape[0]))
    else:
        return _compose_operators(op1, op2)

def _reverse_multiply_operators(op1, op2):
    # i.e. op2 * op1
    assert num._is_number(op2)
    return _compose_operators(
        _ScalarMultiplicationOperator(op1.typecode())(op2, op1.shape[0]),
        op1)

def _call_operator(op1, op2):
    try:
        temp = num.zeros((op1.shape[0],), op2.typecode())
        op1.apply(op2, temp)
        return temp
    except TypeError:
        # attempt applying a real operator to a complex problem
        temp_r = num.zeros((op1.shape[0],), num.Float)
        temp_i = num.zeros((op1.shape[0],), num.Float)
        op1.apply(op2.real, temp_r)
        op1.apply(op2.imaginary, temp_i)
        return temp_r + 1j*temp_i




def _add_operator_behaviors():
    def get_returner(value):
        # This routine is necessary since we don't want the lambda in
        # the top-level scope, whose variables change.
        return lambda self: value

    for tc in num.TYPECODES:
        Operator(tc).__neg__ = _neg_operator
        Operator(tc).__add__ = _add_operators
        Operator(tc).__sub__ = _subtract_operators
        Operator(tc).__mul__ = _multiply_operators
        Operator(tc).__rmul__ = _reverse_multiply_operators
        Operator(tc).__call__ = _call_operator
        Operator(tc).typecode = get_returner(tc)




_add_operator_behaviors()






