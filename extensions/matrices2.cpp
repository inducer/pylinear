#include "matrices.h"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeMatrixType(ublas::coordinate_matrix<ValueType>(), "SparseBuildMatrix", python_eltypename);
}




void pylinear_matrices_part2()
{
  EXPOSE_ALL_TYPES;
}


