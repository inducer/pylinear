#include "matrices.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeMatrixType(ublas::compressed_matrix<ValueType, ublas::column_major>(), "SparseExecuteMatrix", python_eltypename);
}




void pylinear_matrices_part3()
{
  EXPOSE_ALL_TYPES;
}



