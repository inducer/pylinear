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
      VectorTypeX &x,
      const VectorExpressionB &b, double tol, unsigned max_iterations)
  {
    typedef 
      typename MatrixExpression::value_type
      v_t;

    if (A().size1() != A().size2())
      throw std::runtime_error("cg: A is not quadratic");

    identity_matrix<v_t> preconditioner(A.size1(),A.size2());

    // typed up from J.R. Shewchuck, 
    // An Introduction to the Conjugate Gradient Method
    // Without the Agonizing Pain, Edition 1 1/4 [8/1994]
    // Appendix B4
    unsigned iterations = 0;
    vector<v_t> residual(b-prod(A,x));
    vector<v_t> d = prod(preconditioner, residual);

    v_t delta_new = inner_prod(conj(residual), d);
    v_t delta_0 = delta_new;

    while (iterations < max_iterations && 
        absolute_value(delta_new) > tol*tol * absolute_value(delta_0))
    {
      vector<v_t> q = prod(A, d);
      v_t alpha = delta_new / inner_prod(conj(d), q);

      x += alpha * d;

      if (iterations % 50 == 0)
        residual = b - prod(A,x);
      else
        residual -= alpha*q;

      vector<v_t> s = prod(preconditioner, residual);
      v_t delta_old = delta_new;
      delta_new = inner_prod(conj(residual), s);

      v_t beta = delta_new / delta_old;
      d = s + beta * d;

      iterations++;
    }

    if ( iterations == max_iterations)
      throw std::runtime_error("cg failed to converge");
  }
}




#endif
