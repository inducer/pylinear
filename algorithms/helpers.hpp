#ifndef HEADER_SEEN_HELPERS_HPP
#define HEADER_SEEN_HELPERS_HPP



#include <complex>
#include "managed_adaptors.hpp"
#include "generic_ublas.hpp"
#include <boost/numeric/ublas/vector.hpp>




namespace helpers {
namespace ublas = boost::numeric::ublas;




// decomplexify ---------------------------------------------------------------
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




// complexify -----------------------------------------------------------------
template <typename T>
struct complexify
{
  typedef std::complex<T> type;
};

template <typename ELT>
struct complexify<std::complex<ELT> >
{
  typedef std::complex<ELT> type;
};




// isComplex ------------------------------------------------------------------
template <typename T>
inline bool isComplex(const T &)
{
  return false;
}




template <typename T2>
inline bool isComplex(const std::complex<T2> &)
{
  return true;
}




// isSparse -------------------------------------------------------------------
template <typename T>
inline bool isSparse(const T &)
{
  return true;
}




template <typename T2>
inline bool isSparse(const ublas::matrix<T2> &)
{
  return false;
}




template <typename T2>
inline bool isSparse(const ublas::vector<T2> &)
{
  return false;
}




// isHermitian ----------------------------------------------------------------
template <typename MatrixType>
inline bool isHermitian(const MatrixType &)
{
  return false;
}




template <typename T>
inline bool isHermitian(const managed_symmetric_adaptor<T> &)
{
  return true;
}




template <typename T>
inline bool isHermitian(const managed_hermitian_adaptor<T> &)
{
  return true;
}





// isCoordinateMatrix ---------------------------------------------------------
template <typename MatrixType>
inline bool isCoordinateMatrix(const MatrixType &)
{
  return false;
}




template <typename T>
inline bool isCoordinateMatrix(const ublas::coordinate_matrix<T> &)
{
  return true;
}




// conjugate ------------------------------------------------------------------
template <typename T>
inline T conjugate(const T &x)
{
  return x;
}




template <typename T2>
inline std::complex<T2> conjugate(const std::complex<T2> &x)
{
  return conj(x);
}




// conjugate_if ---------------------------------------------------------------
template <typename T>
inline T conjugate_if(bool do_it, const T &x)
{
  return x;
}




template <typename T2>
inline std::complex<T2> conjugate_if(bool do_it, const std::complex<T2> &x)
{
  return do_it ? conj(x) : x;
}




// absolute_value -------------------------------------------------------------
template <typename T>
inline T absolute_value(const T &x)
{
  return fabs(x);
}




template <typename T2>
inline T2 absolute_value(const std::complex<T2> &x)
{
  return sqrt(norm(x));
}




// fill_matrix ----------------------------------------------------------------
template <typename MatrixType>
void fill_matrix(MatrixType &me, 
    const typename MatrixType::value_type &value)
{
  generic_ublas::matrix_iterator<MatrixType>
    first = generic_ublas::begin(me), last = generic_ublas::end(me);
  
  while (first != last)
    *first++ = value;
}




// end namespaces -------------------------------------------------------------
}




#endif
