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




#include "array.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeVectorType(ublas::vector<ValueType>(), "Vector", python_eltypename);
}




void pylinear_expose_vector()
{
  EXPOSE_ALL_TYPES;
}

