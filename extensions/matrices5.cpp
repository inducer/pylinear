#include "matrices.h"




void pylinear_matrices_part5()
{
  exposeMatrixType(managed_symmetric_adaptor<
      ublas::coordinate_matrix<double> >(), 
      "SparseSymmetricBuildMatrix", "Float64");
  exposeMatrixType(managed_hermitian_adaptor<
      ublas::coordinate_matrix<std::complex<double> > >(), 
      "SparseHermitianBuildMatrix", "Complex64");
}


