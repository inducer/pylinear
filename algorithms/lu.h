#ifndef HEADER_SEEN_LU_H
#define HEADER_SEEN_LU_H




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "helpers.h"



namespace lu
{
  using namespace boost::numeric::ublas;
  using namespace helpers;





  template <typename MatrixExpression, typename DesiredResultL, typename DesiredResultU>
  std::pair<DesiredResultL *, DesiredResultU *> lu(const MatrixExpression &a)
  {
    if (a().size1() != a().size2())
      throw std::runtime_error("cholesky: A is not quadratic");

    std::auto_ptr<DesiredResultL> result_l(new DesiredResultL(a().size1(), a().size2()));
    std::auto_ptr<DesiredResultU> result_u(new DesiredResultU(a().size1(), a().size2()));

    DesiredResultL &l = *result_l;
    DesiredResultU &u = *result_u;

    typedef 
      typename MatrixExpression::size_type
      size_type;
    typedef 
      typename MatrixExpression::value_type
      value_type;

    // FIXME: this needs to be replaced by at least a pivoting version,
    // or we should drill open umfpack
    for (size_type col = 0; col < a().size2(); col++) 
    {
      value_type sum;

      // elements above diagonal --> r
      for (size_type row = 0; row < col; row++) 
      {
        sum = 0;
        for (size_type i = 0;i < row;i++)
          sum += value_type(l(row,i))*value_type(u(i,col));
        // the corresponding l element is 1 anyway
        u(row,col) = a()(row,col)-sum;
      }

      // elements on diagonal
      sum = 0;
      for (size_type i = 0;i<col;i++)
        sum += value_type(l(col,i))*value_type(u(i,col));

      l(col,col) = 1;
      u(col,col) = a()(col,col)-sum;

      //elements under diagonal --> l
      for (size_type row = col+1; row < a().size1(); row++) 
      {
        sum = 0;
        for (size_type i = 0; i < col; i++)
          sum += value_type(l(row,i))*value_type(u(i,col));
        l(row,col) = (value_type(a()(row,col))-sum)/value_type(u(col,col));
      }
    }

    return std::make_pair(result_l.release(), result_u.release());
  }
}




#endif
