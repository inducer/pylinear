#ifndef HEADER_SEEN_LU_H
#define HEADER_SEEN_LU_H




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/tuple/tuple.hpp>
#include "helpers.h"
#include <complex>



namespace lu
{
  namespace ublas = boost::numeric::ublas;
  using namespace boost::numeric::ublas;
  using namespace helpers;
  using namespace boost::tuples;





  inline double magnitude(double x)
  {
    return fabs(x);
  }




  inline double magnitude(std::complex<double> x)
  {
    return std::max(fabs(real(x)), fabs(imag(x)));
  }




  /** Postcondition P*A = L*U.
   */
  template <typename MatrixExpression, typename DesiredResultL, typename DesiredResultU>
  boost::tuple<DesiredResultL *, DesiredResultU *, std::vector<unsigned> *, int> lu(const MatrixExpression &a)
  {
    if (a().size1() != a().size2())
      throw std::runtime_error("cholesky: A is not quadratic");

    unsigned n = a().size1();
    int parity = 1;

    std::auto_ptr<DesiredResultL> result_l(new DesiredResultL(n, n));
    std::auto_ptr<DesiredResultU> result_u(new DesiredResultU(n, n));
    std::auto_ptr<std::vector<unsigned> > permut_ptr(new std::vector<unsigned>);
    permut_ptr->resize(n);

    DesiredResultL &l = *result_l;
    DesiredResultU &u = *result_u;
    std::vector<unsigned> &permut = *permut_ptr;

    for (unsigned i = 0; i <n; i++)
      permut[i] = i;

    typedef 
      typename MatrixExpression::size_type
      size_type;
    typedef 
      typename MatrixExpression::value_type
      value_type;

    /* Modified from Numerical Recipes, par. 2.3: Crout with implicit scaling
     */

    // FIXME: investigate optimization using iterators?

    // Implicit scaling -------------------------------------------------------
    // pretend that each row is scaled to have its largest entry be on the order of 1.
    double scale_factor[n];

    for (size_type row = 0; row < n; row++)
    {
      double temp;
      double big = 0;
      big = 0;
      for (size_type col = 0; col < n; col++)
        if ((temp = magnitude(a()(row, col))) > big) 
          big = temp;
      if (big == 0)
        throw std::runtime_error("lu: all-zero column in lu decomposition");
      scale_factor[row] = 1./big;
    }
    
    // Crout ------------------------------------------------------------------
    double temp;
    double biggest;
    size_type biggest_row;
    value_type sum;
    value_type pivot;

    for (size_type col = 0; col < n; col++) 
    {
      // elements above diagonal --> r
      for (size_type row = 0; row < col; row++) 
      {
        sum = a()(permut[row], col);
        for (size_type i = 0;i < row;i++)
          sum -= value_type(l(row, i))*value_type(u(i, col));

        u(row, col) = sum;

        /*
         * vectorizing does not make this faster:
        u(row, col) = a()(permut[row], col) - inner_prod(
            project(ublas::row(l, row), range(0, row)),
            project(column(u, col), range(0, row))
            );
            */
      }

      biggest = 0;
      biggest_row = 0;

      // element on diagonal --> first into l, then into u
      sum = a()(permut[col], col);
      for (size_type i = 0; i < col; i++)
        sum -= value_type(l(col, i))*value_type(u(i, col));
      l(col, col) = sum;

      if ((temp = scale_factor[permut[col]] * magnitude(sum)) >= biggest)
      {
        biggest = temp;
        biggest_row = col;
      }

      // elements below diagonal --> l
      for (size_type row = col + 1; row < n; row++) 
      {
        sum = a()(permut[row], col);
        for (size_type i = 0; i < col; i++)
          sum -= value_type(l(row, i))*value_type(u(i, col));
        l(row, col) = sum;

        if ((temp = scale_factor[permut[row]] * magnitude(sum)) >= biggest)
        {
          biggest = temp;
          biggest_row = row;
        }
      }

      if (biggest == 0)
        throw std::runtime_error("lu: matrix is singular");

      if (biggest_row != col)
      {
        // swap access array for a
        std::swap(permut[biggest_row], permut[col]);

        // swap rows of l accordingly
        swap(
            project(ublas::row(l, col), range(0, col+1)), 
            project(ublas::row(l, biggest_row), range(0, col+1))
            );

        parity *= -1;
      }
      pivot = u(col, col) = l(col, col);
      l(col, col) = 1;

      project(column(l, col), range(col+1, n)) *= 1. / pivot;
    }

    return make_tuple(result_l.release(), result_u.release(), permut_ptr.release(), parity);
  }
}




#endif
