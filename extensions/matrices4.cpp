#include "matrices.h"




void pylinear_matrices_part4()
{
  exposeMatrixType(managed_symmetric_adaptor<
      ublas::compressed_matrix<double, ublas::column_major> >(), 
      "SparseSymmetricExecuteMatrix", "Float64");
  exposeMatrixType(managed_hermitian_adaptor<
      ublas::compressed_matrix<std::complex<double>, ublas::column_major > >(), 
      "SparseHermitianExecuteMatrix", "Complex64");
}

