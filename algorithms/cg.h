#ifndef HEADER_SEEN_CG_H
#define HEADER_SEEN_CG_H




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "helpers.h"
#include "matrix_operator.h"



namespace cg
{
  using namespace boost::numeric::ublas;
  using namespace helpers;




  template <typename MatrixExpression, typename PreconditionerExpression,
  typename VectorTypeX, typename VectorExpressionB>
  void solveCG(
      const MatrixExpression &A, 
      const PreconditionerExpression &preconditioner, 
      VectorTypeX &x,
      const VectorExpressionB &b, double tol, unsigned max_iterations, 
      unsigned *iteration_count = NULL, unsigned debug_level = 0)
  {
    typedef 
      typename MatrixExpression::value_type
      v_t;

    if (A().size1() != A().size2())
      throw std::runtime_error("cg: A is not quadratic");

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

      if (debug_level && iterations % 20 == 0)
        std::cout << delta_new << std::endl;

      iterations++;
    }

    if ( iterations == max_iterations)
      throw std::runtime_error("cg failed to converge");

    if (iteration_count)
      *iteration_count = iterations;
  }




  template <typename ValueType>
  class cg_matrix_operator : public iterative_solver_matrix_operator<ValueType>
  {
    typedef matrix_operator<ValueType> mop_type;
    const mop_type &m_matrix;
    const mop_type &m_preconditioner;

    typedef
      iterative_solver_matrix_operator<ValueType>
      super;

  public:
    typedef 
      typename super::vector_type
      vector_type;

    cg_matrix_operator(const mop_type &mat, const mop_type &precon, unsigned maxit, double tol)
    : super(maxit, tol), m_matrix(mat), m_preconditioner(precon)
    { 
      if (mat.size1() != mat.size2())
        throw std::runtime_error("cg: matrix has to be quadratic (and sym. pos. def.) to work with cg");
    }

    unsigned size1() const
    {
      return m_matrix.size2();
    }
    unsigned size2() const
    {
      return m_matrix.size1();
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      cg::solveCG(m_matrix, m_preconditioner, after, before, m_tolerance,
          m_maxIterations, const_cast<unsigned *>(&m_lastIterationCount), m_debugLevel);
    }
  };
}




#endif
