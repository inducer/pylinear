#ifndef HEADER_SEEN_HELPERS_H
#define HEADER_SEEN_HELPERS_H



#include <complex>




namespace helpers {
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
  return norm(x);
}




// end namespaces -------------------------------------------------------------
}




#endif
