#include "matrices.h"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeVectorType(ublas::vector<ValueType>(), "Vector", python_eltypename);
  exposeMatrixType(ublas::matrix<ValueType>(), "Matrix", python_eltypename);
}




void pylinear_matrices_part2();
void pylinear_matrices_part3();
void pylinear_matrices_part4();
void pylinear_matrices_part5();




BOOST_PYTHON_MODULE(matrices_internal)
{
  enum_<SupportedElementTypes>("SupportedElementTypes")
    .value("Float64", Float64)
    .value("Complex64", Complex64)
    .export_values();

  EXPOSE_ALL_TYPES;
  pylinear_matrices_part2();
  pylinear_matrices_part3();
  pylinear_matrices_part4();
  pylinear_matrices_part5();
}

