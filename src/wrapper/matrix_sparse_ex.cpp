//
// Copyright (c) 2004-2006
// Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#define PYLINEAR_NO_UFUNCS
#include "array.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeMatrixType(ublas::compressed_matrix<
                   ValueType, ublas::column_major, 0, 
                   ublas::unbounded_array<int> >(), 
                   "SparseExecuteMatrix", python_eltypename);
}




void pylinear_expose_sparse_ex()
{
  EXPOSE_ALL_TYPES;
}



