#include "matrices.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeVectorType(ublas::vector<ValueType>(), "Vector", python_eltypename);
}




void pylinear_expose_vectors()
{
  EXPOSE_ALL_TYPES;
}

