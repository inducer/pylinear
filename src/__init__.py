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




"""PyLinear -- linear algebra in Python.

See http://news.tiker.net/software/pylinear for details.
"""

import _operation as _op

def version():
    """Return a 3-tuple with the PyLinear version."""
    return (0,92,0)

has_blas = _op.has_blas
has_lapack = _op.has_lapack
has_arpack = _op.has_arpack
has_umfpack = _op.has_umfpack
has_daskr = _op.has_daskr
