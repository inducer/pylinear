#ifndef HEADER_SEEN_CHOLESKY_HPP
#define HEADER_SEEN_CHOLESKY_HPP




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "helpers.hpp"



namespace cholesky
{
  using namespace boost::numeric::ublas;
  using namespace helpers;





  inline double conj(double x)
  {
    return x;
  }




  inline double real(double x)
  {
    return x;
  }




  /** This routine only uses the lower-left half of A.
   */
  template <typename MatrixExpression, typename DesiredResult>
  DesiredResult *cholesky(const MatrixExpression &A)
  {
    if (A().size1() != A().size2())
      throw std::runtime_error("cholesky: A is not quadratic");

    std::auto_ptr<DesiredResult> result(new DesiredResult(A().size1(), A().size2()));
    DesiredResult &L = *result;

    typedef 
      typename MatrixExpression::size_type
      size_type;
    typedef 
      typename MatrixExpression::value_type
      value_type;

    // FIXME: this is a quick prototype

    for (size_type col = 0; col < A().size2(); col++) 
    {
      // determine diagonal element
      value_type sum = A()(col,col);
      for (size_type row = 0; row < col; row++) 
      {
        value_type value = L(col,row);
        sum -= value*conj(value);
      }
      if (real(sum) < 0 && !helpers::isComplex(value_type()))
        throw std::runtime_error("cholesky: matrix not positive definite");

      L(col,col) = sqrt(sum);

      // determine off-diagonal elements
      for (size_type row = col+1; row < A().size2(); row++) 
      {
        sum = A()(row,col);
        for (size_type i = 0; i < col; i++)
        {
          // FIXME : casts needed due to bug in sparse uBLAS
          sum -= value_type(L(row,i))*value_type(conj(value_type(L(col,i))));
        }
        L(row,col) = sum/value_type(L(col,col));
      }
    }

    return result.release();
  }
}




#endif
