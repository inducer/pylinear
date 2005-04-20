#include "array.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeMatrixType(ublas::matrix<ValueType>(), "Matrix", python_eltypename);
}




void pylinear_expose_dense()
{
  EXPOSE_ALL_TYPES;
}



