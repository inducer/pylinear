#include <boost/python.hpp>
#include <cg.h>
#include "meta.h"



using namespace boost::python;




// umfpack --------------------------------------------------------------------
struct umfpack_algorithm_type
{
  template <typename MatrixType>
  static void expose(MatrixType)
  {
    typedef 
      typename get_corresponding_vector_type<MatrixType>::type 
      vector_type;

    def("cg", cg::cg<MatrixType, vector_type, vector_type>);
  }
};




// cg -------------------------------------------------------------------------
struct cg_algorithm_type
{
  template <typename MatrixType>
  static void expose(MatrixType)
  {
    typedef 
      typename get_corresponding_vector_type<MatrixType>::type 
      vector_type;

    def("cg", cg::cg<MatrixType, vector_type, vector_type>);
  }
};




// generic instantiation infrastructure ---------------------------------------
template <typename AlgorithmType, typename ValueType>
void instantiateForAllSimpleTypes(AlgorithmType, ValueType)
{
  AlgorithmType::expose(ublas::matrix<ValueType>());
  AlgorithmType::expose(ublas::compressed_matrix<ValueType>());
  AlgorithmType::expose(ublas::coordinate_matrix<ValueType>());
}




template <typename AlgorithmType>
void instantiateForAllMatrices(AlgorithmType)
{
  instantiateForAllSimpleTypes(AlgorithmType(), double());
  instantiateForAllSimpleTypes(AlgorithmType(), std::complex<double>());

  AlgorithmType::expose(managed_symmetric_adaptor<
      ublas::compressed_matrix<double> >());
  AlgorithmType::expose(managed_symmetric_adaptor<
      ublas::coordinate_matrix<double> >());
  AlgorithmType::expose(managed_hermitian_adaptor<
      ublas::compressed_matrix<std::complex<double> > >());
  AlgorithmType::expose(managed_hermitian_adaptor<
      ublas::coordinate_matrix<std::complex<double> > >());
}




// main -----------------------------------------------------------------------
BOOST_PYTHON_MODULE(algorithms)
{
  instantiateForAllMatrices(cg_algorithm_type());
}
