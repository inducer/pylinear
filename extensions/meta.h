#ifndef HEADER_SEEN_META_H
#define HEADER_SEEN_META_H




#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "managed_adaptors.h"




using namespace boost;
namespace ublas = boost::numeric::ublas;




// metaprogramming helpers ----------------------------------------------------
namespace value_type_promotion
{
  using std::complex;

  template <typename A, typename B>
  struct bigger_type
  {
  };

  template <typename A>
  struct bigger_type<A,A> { typedef A type; };

  template <> struct bigger_type<float, double> { typedef double type; };
  template <> struct bigger_type<float, complex<float> > { typedef complex<float> type; };
  template <> struct bigger_type<float, complex<double> > { typedef complex<double> type; };

  template <> struct bigger_type<double, complex<float> > { typedef complex<double> type; };
  template <> struct bigger_type<double, complex<double> > { typedef complex<double> type; };

  template <> struct bigger_type<complex<float>, complex<double> > { typedef complex<double> type; };

  // and the other way around
  template <> struct bigger_type<double, float> { typedef double type; };
  template <> struct bigger_type<complex<float>, float> { typedef complex<float> type; };
  template <> struct bigger_type<complex<double>, float> { typedef complex<double> type; };

  template <> struct bigger_type<complex<float>, double> { typedef complex<double> type; };
  template <> struct bigger_type<complex<double>, double> { typedef complex<double> type; };

  template <> struct bigger_type<complex<double>, complex<float> > { typedef complex<double> type; };
}




// is_vector
template <typename UblasType>
struct is_vector { typedef mpl::false_ type; };

template <typename ValueType>
struct is_vector<ublas::vector<ValueType> > { typedef mpl::true_ type; };
template <typename WrappedVector>
struct is_vector<ublas::vector_slice<WrappedVector> > { typedef mpl::true_ type; };





// decomplexify
template <typename T>
struct decomplexify
{
  typedef T type;
};

template <typename ELT>
struct decomplexify<std::complex<ELT> >
{
  typedef ELT type;
};




// get_computation_result_type
template <typename MatrixType>
struct get_computation_result_type 
{ typedef MatrixType type; };




// change_value_type
template <typename MatrixType, typename NewValueType>
struct change_value_type { };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::matrix<OldValueType>, NewValueType>
{ typedef ublas::matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::sparse_matrix<OldValueType>, NewValueType>
{ typedef ublas::sparse_matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::coordinate_matrix<OldValueType>, NewValueType>
{ typedef ublas::coordinate_matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::compressed_matrix<OldValueType>, NewValueType>
{ typedef ublas::compressed_matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::vector<OldValueType>, NewValueType>
{ typedef ublas::vector<NewValueType> type; };

template <typename ContainedMatrixType, typename NewValueType>
struct change_value_type<managed_symmetric_adaptor<ContainedMatrixType>, NewValueType>
{ 
  typedef 
    managed_symmetric_adaptor<
    typename change_value_type<ContainedMatrixType, NewValueType>::type> type; 
};

template <typename ContainedMatrixType, typename NewValueType>
struct change_value_type<managed_hermitian_adaptor<ContainedMatrixType>, NewValueType>
{ 
  typedef 
    managed_hermitian_adaptor<
    typename change_value_type<ContainedMatrixType, NewValueType>::type> type; 
};




// get_corresponding_vector_type
template <typename MatrixType>
struct get_corresponding_vector_type 
{ typedef ublas::vector<typename MatrixType::value_type> type; };




#endif
