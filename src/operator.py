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
PyLinear's Python module for matrix-free methods.
"""




import pylinear
import pylinear.array as num
import pylinear.computation as comp
import pylinear._operation as _op

# operator parameterized types ------------------------------------------------
Operator = num.ParameterizedType(
  "MatrixOperator", _op.__dict__)
IdentityOperator = num.ParameterizedType(
  "IdentityMatrixOperator", _op.__dict__)
ScalarMultiplicationOperator = num.ParameterizedType(
  "ScalarMultiplicationMatrixOperator", _op.__dict__)

class _MatrixOperatorParameterizedType(object):
    def is_a(self, instance):
        # FIXME
        raise NotImplementedError

    def __call__(self, dtype, flavor):
        # FIXME
        raise NotImplementedError

    def make(self, matrix):
        return _op.makeMatrixOperator(matrix)
MatrixOperator = _MatrixOperatorParameterizedType()

class _CGParameterizedType(num.ParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.dtype, w)
        if matrix_op.dtype is not precon_op.dtype:
            raise TypeError, "matrix_op and precon_op must have matching dtypes"
        return self.TypeDict[matrix_op.dtype](matrix_op, precon_op, max_it, tolerance)
    
CGOperator = _CGParameterizedType("CGMatrixOperator", _op.__dict__)

class _BiCGSTABParameterizedType(num.ParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.dtype, w)
        if matrix_op.dtype is not precon_op.dtype:
            raise TypeError, "matrix_op and precon_op must have matching dtypes"
        return self.TypeDict[matrix_op.dtype](matrix_op, precon_op, max_it, tolerance)
    
BiCGSTABOperator = _BiCGSTABParameterizedType(
    "BiCGSTABMatrixOperator", _op.__dict__)

if pylinear.has_umfpack():
    class _UMFPACKParameterizedType(num.ParameterizedType):
        def make(self, matrix):
            matrix.complete_index1_data()
            return self.TypeDict[matrix.dtype](matrix)

    UMFPACKOperator = _UMFPACKParameterizedType("UMFPACKMatrixOperator", 
                                                        _op.__dict__)

class _LUInverseOperator:
    def __init__(self, l, u, perm):
        assert l.shape[0] == l.shape[1]
        assert u.shape[0] == u.shape[1]
        assert l.shape[0] == u.shape[0]

        self.L = l
        self.U = u
        self.P = num.permutation_matrix(from_indices=perm)

    def size1(self):
        return self.L.shape[0]
    
    def size2(self):
        return self.L.shape[1]

    def apply(self, before, after):
        after[:] = self.U.solve_upper(
                self.L.solve_lower(
                    self.P*before))

class _LUInverseOperatorFloat64(_LUInverseOperator, _op.MatrixOperatorFloat64):
    def __init__(self, l, u, perm):
        _LUInverseOperator.__init__(self, l, u, perm)
        _op.MatrixOperatorFloat64.__init__(self)

class _LUInverseOperatorComplex64(_LUInverseOperator, _op.MatrixOperatorComplex64):
    def __init__(self, l, u, perm):
        _LUInverseOperator.__init__(self, l, u, perm)
        _op.MatrixOperatorComplex64.__init__(self)

class _LUInverseParameterizedType(num.ParameterizedType):
    def make(self, *args):
        if len(args) == 3:
            l, u, perm = args
        elif len(args) == 1:
            l, u, perm, sign = comp.lu(args[0])
        else:
            raise TypeError, "Invalid number of arguments"

        return self.TypeDict[l.dtype](l, u, perm)

LUInverseOperator = _LUInverseParameterizedType("_LUInverseOperator", 
        globals())

class _SSORPreconditioner:
    def __init__(self, mat, omega=1):
        # mat needs to be symmetric
        assert mat.shape[0] == mat.shape[1]

        l = num.tril(mat, -1)
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

class _SSORPreconditionerParameterizedType(num.ParameterizedType):
    def make(self, mat, *args, **kwargs):
        return num.ParameterizedType.make(
            self, mat.dtype, mat, *args, **kwargs)

SSORPreconditioner = _SSORPreconditionerParameterizedType(
    "_SSORPreconditioner", globals())


# operator operators ----------------------------------------------------------
_SumOfOperators = num.ParameterizedType(
  "SumOfMatrixOperators", _op.__dict__)
_ScalarMultiplicationOperator = num.ParameterizedType(
  "ScalarMultiplicationMatrixOperator", _op.__dict__)
_CompositeOfOperators = num.ParameterizedType(
  "CompositeMatrixOperator", _op.__dict__)




def _neg_operator(op):
    return _compose_operators(
        _ScalarMultiplicationOperator(op.dtype)(-1, op.shape[0]),
        op)

def _add_operators(op1, op2):
    return _SumOfOperators(op1.dtype)(op1, op2)

def _subtract_operators(op1, op2):
    return _add_operators(op1, _neg_operator(op2))

def _compose_operators(outer, inner):
    return _CompositeOfOperators(outer.dtype)(outer, inner)

def _multiply_operators(op1, op2):
    if num._is_number(op2):
        return _compose_operators(
            op1,
            _ScalarMultiplicationOperator(op1.dtype)(op2, op1.shape[0]))
    else:
        return _compose_operators(op1, op2)

def _reverse_multiply_operators(op1, op2):
    # i.e. op2 * op1
    assert num._is_number(op2)
    return _compose_operators(
        _ScalarMultiplicationOperator(op1.dtype)(op2, op1.shape[0]),
        op1)

def _call_operator(op1, op2):
    try:
        temp = num.zeros((op1.shape[0],), op2.dtype)
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

    for dtype in num.DTYPES:
        Operator(dtype).__neg__ = _neg_operator
        Operator(dtype).__add__ = _add_operators
        Operator(dtype).__sub__ = _subtract_operators
        Operator(dtype).__mul__ = _multiply_operators
        Operator(dtype).__rmul__ = _reverse_multiply_operators
        Operator(dtype).__call__ = _call_operator
        Operator(dtype).typecode = get_returner(dtype)
        Operator(dtype).dtype = property(get_returner(dtype))




_add_operator_behaviors()
