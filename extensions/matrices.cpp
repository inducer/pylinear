#include "matrices.hpp"




template <typename ValueType>
static void exposeAll(ValueType, const std::string &python_eltypename)
{
  exposeMatrixType(ublas::matrix<ValueType>(), "Matrix", python_eltypename);
}




void pylinear_expose_sym_build();
void pylinear_expose_sym_ex();
void pylinear_expose_sparse_build();
void pylinear_expose_sparse_ex();
void pylinear_expose_vectors();




BOOST_PYTHON_MODULE(_matrices)
{
  enum_<SupportedElementTypes>("SupportedElementTypes")
    .value("Float64", Float64)
    .value("Complex64", Complex64)
    .export_values();

  EXPOSE_ALL_TYPES;

  //pylinear_expose_sym_build();
  //pylinear_expose_sym_ex();
  pylinear_expose_sparse_build();
  pylinear_expose_sparse_ex();
  pylinear_expose_vectors();
}

