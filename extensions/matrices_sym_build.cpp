#include "matrices.hpp"




void pylinear_expose_sym_build()
{
  exposeMatrixType(managed_symmetric_adaptor<
      ublas::coordinate_matrix<double> >(), 
      "SparseSymmetricBuildMatrix", "Float64");
  exposeMatrixType(managed_hermitian_adaptor<
      ublas::coordinate_matrix<std::complex<double> > >(), 
      "SparseHermitianBuildMatrix", "Complex64");
}


