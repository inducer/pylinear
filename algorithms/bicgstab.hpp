#ifndef HEADER_SEEN_BICGSTAB_HPP
#define HEADER_SEEN_BICGSTAB_HPP




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "helpers.hpp"
#include "matrix_operator.hpp"



namespace bicgstab
{
  using namespace boost::numeric::ublas;
  using namespace helpers;




  template <typename MatrixExpression, typename PreconditionerExpression,
  typename VectorTypeX, typename VectorExpressionB>
  void solveBiCGSTAB(const MatrixExpression &A, 
	       const PreconditionerExpression &preconditioner, 
	       VectorTypeX &start_guess,
	       const VectorExpressionB &b, double tol, unsigned max_iterations, 
	       unsigned *iteration_count = NULL, unsigned debug_level = 0)
  {
    typedef 
      typename MatrixExpression::value_type
      v_t;

    if (A().size1() != A().size2())
      throw std::runtime_error("bicgstab: A is not quadratic");

    // typed up from Figure 2.10 of 
    // Templates for the Solution of Linear Systems: 
    // Building Blocks for Iterative Methods
    // (Barrett, R., M. Berry, T. F. Chan, et al.)

    // THIS ALGORITHM WILL PROBABLY NOT WORK ON COMPLEX NUMBERS!

    unsigned iterations = 0;

    // "next" refers to i
    // "" refers to i-1
    // "last" refers to i-2

    v_t rho, last_rho, next_alpha, alpha, next_omega, omega;
    vector<v_t> next_p, p, next_v, v, next_x, x(start_guess);

    vector<v_t> r(b-prod(A,x)), next_r;
    vector<v_t> r_tilde(r);
    v_t initial_residual = norm2(r);

    while (iterations < max_iterations)
    {
      rho = inner_prod(r_tilde, r);
      if (rho == 0)
	throw std::runtime_error("bicgstab failed, rho == 0");
      if (iterations == 0)
	next_p = r;
      else
	{
	  double beta = (rho/last_rho)*(alpha/omega);
	  next_p = r + beta*(p-omega*v);
	}

      vector<v_t> p_hat = prod(preconditioner, next_p);
      next_v = prod(A, p_hat);
      next_alpha = rho/inner_prod(r_tilde, next_v);
      vector<v_t> s = r - next_alpha*next_v;

      if (norm2(s) < tol * initial_residual)
	{
	  next_x = x + next_alpha*p_hat;
	  break;
	}

      vector<v_t> s_hat = prod(preconditioner, s);
      vector<v_t> t = prod(A, s_hat);
      next_omega = inner_prod(t, s)/inner_prod(t, t);
      next_x = x + next_alpha * p_hat + next_omega * s_hat;
      next_r = s - next_omega * t;

      if (next_omega == 0)
	throw std::runtime_error("bicgstab failed, omega == 0");

      if (norm2(r) < tol * initial_residual)
	break;

      last_rho = rho;
      p = next_p;
      v = next_v;
      x = next_x;
      r = next_r;
      alpha = next_alpha;
      omega = next_omega;

      if (debug_level && iterations % 20 == 0)
        std::cout << delta_new << std::endl;
      iterations++;
    }

    if ( iterations == max_iterations)
      throw std::runtime_error("bicgstab failed to converge");

    if (iteration_count)
      *iteration_count = iterations;
  }




  template <typename ValueType>
  class bicgstab_matrix_operator : public iterative_solver_matrix_operator<ValueType>
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
        throw std::runtime_error("bicgstab: matrix has to be quadratic (and sym. pos. def.) to work with cg");
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

      cg::solveBiCGSTAB(m_matrix, 
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
