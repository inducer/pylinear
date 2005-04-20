#include "array.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeMatrixType(ublas::coordinate_matrix<ValueType>(), "SparseBuildMatrix", python_eltypename);
}




void pylinear_expose_sparse_build()
{
  EXPOSE_ALL_TYPES;
}


