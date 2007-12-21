//
// Copyright (c) 2004-2006
// Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#ifndef HEADER_SEEN_CG_HPP
#define HEADER_SEEN_CG_HPP




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "helpers.hpp"
#include "matrix_operator.hpp"



namespace cg
{
  using namespace boost::numeric::ublas;
  using namespace helpers;




  template <typename MatrixExpression, typename PreconditionerExpression,
  typename VectorTypeX, typename VectorExpressionB>
  void solveCG(const MatrixExpression &A, 
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

    // typed up from J.R. Shewchuk, 
    // An Introduction to the Conjugate Gradient Method
    // Without the Agonizing Pain, Edition 1 1/4 [8/1994]
    // Appendix B3
    unsigned iterations = 0;
    vector<v_t> residual(b-prod(A,x));
    vector<v_t> d = prod(preconditioner, residual);

    v_t delta_new = inner_prod(residual, conj(d));
    v_t delta_0 = delta_new;

    while (iterations < max_iterations)
    {
      vector<v_t> q = prod(A, d);
      v_t alpha = delta_new / inner_prod(d, conj(q));

      x += alpha * d;
      bool calculate_real_residual = iterations % 50 == 0 || \
	absolute_value(delta_new) < tol*tol * absolute_value(delta_0);

      if (calculate_real_residual)
        residual = b - prod(A, x);
      else
        residual -= alpha*q;

      vector<v_t> s = prod(preconditioner, residual);
      v_t delta_old = delta_new;
      delta_new = inner_prod(residual, conj(s));

      if (calculate_real_residual && absolute_value(delta_new) < tol*tol * absolute_value(delta_0))
      {
	// Only terminate the loop on the basis of a "real" residual.
	break;
      }

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
      super::apply(before, after);

      after.clear();
      cg::solveCG(m_matrix, 
		  m_preconditioner, 
		  after, 
		  before, 
		  this->m_tolerance,
		  this->m_maxIterations, 
		  const_cast<unsigned *>(&this->m_lastIterationCount), 
		  this->m_debugLevel);
    }
  };
}




#endif
