#ifndef HEADER_SEEN_CG_H
#define HEADER_SEEN_CG_H




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "helpers.h"



namespace cg
{
  using namespace boost::numeric::ublas;
  using namespace helpers;




  template <typename MatrixExpression, typename VectorTypeX, typename VectorExpressionB>
  static void cg(
      const MatrixExpression &A, 
      const VectorTypeX &x,
      const VectorExpressionB &b, double tol)
  {
    typedef 
      typename MatrixExpression::value_type
      v_t;

    if (A().size1() != A().size2())
      throw std::runtime_error("cg: A is not quadratic");

    vector<v_t> residual(prod(A,x)-b);
    vector<v_t> u(residual);

    // FIXME: use suitable C
    identity_matrix<v_t> C(A.size1(),A.size2());
    vector<v_t> h(prod(C, residual));
    vector<v_t> d(prod(C, residual));

    v_t residual_dot_h = inner_prod(conj(residual), h);
    while (absolute_value(residual_dot_h) >= tol)
    {
      vector<v_t> Ad = prod(A, d);
      v_t alpha = residual_dot_h / inner_prod(conj(d), Ad);
      u -= alpha * d;
      residual -= alpha * Ad;
      h = prod(C, residual);

      v_t new_residual_dot_h = inner_prod(conj(residual), h); 
      v_t beta = new_residual_dot_h / residual_dot_h;
      d = h + beta * d;
    }
  }
}




#endif
