#ifndef HEADER_SEEN_META_H
#define HEADER_SEEN_META_H




#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "managed_adaptors.h"
#include <helpers.h>




using namespace boost;
namespace ublas = boost::numeric::ublas;




// typecode support -----------------------------------------------------------
enum SupportedElementTypes {
  Float64,
  Complex64,
};




inline SupportedElementTypes getTypeCode(double) { return Float64; }
inline SupportedElementTypes getTypeCode(std::complex<double>) { return Complex64; }




template <typename MatrixType>
inline SupportedElementTypes typecode(const MatrixType &)
{ 
  return getTypeCode(typename MatrixType::value_type());
}




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
struct change_value_type<ublas::compressed_matrix<OldValueType, ublas::column_major>, NewValueType>
{ typedef ublas::compressed_matrix<NewValueType, ublas::column_major> type; };

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




// strip_symmetric_wrappers 
template <typename MatrixType>
struct strip_symmetric_wrappers
{ typedef MatrixType type; };

template <typename MatrixType>
struct strip_symmetric_wrappers<managed_symmetric_adaptor<MatrixType> >
{ typedef MatrixType type; };

template <typename MatrixType>
struct strip_symmetric_wrappers<managed_hermitian_adaptor<MatrixType> >
{ typedef MatrixType type; };




// generic instantiation infrastructure ---------------------------------------
template <typename Exposer, typename ValueType>
static void exposeForAllSimpleTypes(const std::string &python_eltname, const Exposer &exposer, ValueType)
{
  exposer.expose("Matrix" + python_eltname, ublas::matrix<ValueType>());
  exposer.expose("SparseExecuteMatrix" + python_eltname, ublas::compressed_matrix<ValueType, ublas::column_major>());
  exposer.expose("SparseBuildMatrix" + python_eltname, ublas::coordinate_matrix<ValueType>());
}




template <typename Exposer, typename T>
static void exposeForAllMatrices(const Exposer &exposer, T)
{
  exposeForAllSimpleTypes("Float64", exposer, T());

  exposer.expose("SparseSymmetricExecuteMatrixFloat64", managed_symmetric_adaptor<
      ublas::compressed_matrix<T, ublas::column_major> >());
  exposer.expose("SparseSymmetricBuildMatrixFloat64", managed_symmetric_adaptor<
      ublas::coordinate_matrix<T> >());
}




template <typename Exposer, typename T>
static void exposeForAllMatrices(const Exposer &exposer, std::complex<T>)
{
  exposeForAllSimpleTypes("Complex64", exposer, std::complex<T>());

  exposer.expose("SparseHermitianExecuteMatrixComplex64", managed_hermitian_adaptor<
      ublas::compressed_matrix<std::complex<T>, ublas::column_major> >());
  exposer.expose("SparseHermitianBuildMatrixComplex64", managed_hermitian_adaptor<
      ublas::coordinate_matrix<std::complex<T> > >());
}




template <typename Exposer>
static void exposeForAllMatrices(const Exposer &exposer)
{
  exposeForAllMatrices(exposer, double());
  exposeForAllMatrices(exposer, std::complex<double>());
}




template <typename Exposer,typename T>
static void exposeForMatricesConvertibleTo(const Exposer &exposer, T)
{
  exposeForAllMatrices(exposer, T());
}




template <typename Exposer,typename T>
static void exposeForMatricesConvertibleTo(const Exposer &exposer, std::complex<T>)
{
  exposeForAllMatrices(exposer);
}




#endif
