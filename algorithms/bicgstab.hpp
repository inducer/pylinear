#ifndef HEADER_SEEN_BICGSTAB_HPP
#define HEADER_SEEN_BICGSTAB_HPP




#include <stdexcept>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "helpers.hpp"
#include "matrix_operator.hpp"



namespace bicgstab
{
  using namespace boost::numeric::ublas;




  template <typename MatrixExpression, typename PreconditionerExpression,
  typename VectorTypeX, typename VectorExpressionB>
  void solveBiCGSTAB(const MatrixExpression &A, 
	       const PreconditionerExpression &preconditioner, 
	       VectorTypeX &result,
	       const VectorExpressionB &b, double tol, unsigned max_iterations, 
	       unsigned *iteration_count = NULL, unsigned debug_level = 0)
  {
    typedef 
      typename MatrixExpression::value_type
      v_t;

    typedef 
      typename helpers::decomplexify<typename MatrixExpression::value_type>::type
      real_t;

    if (A().size1() != A().size2())
      throw std::runtime_error("bicgstab: A is not quadratic");

    if (debug_level >= 2)
      std::cout << "rhs:" << b << std::endl;

    // typed up from Figure 2.10 of 
    // Templates for the Solution of Linear Systems: 
    // Building Blocks for Iterative Methods
    // (R. Barrett, M. Berry, T. F. Chan, et al.)

    unsigned iterations = 0;

    // "next" refers to i
    // "" refers to i-1
    // "last" refers to i-2

    v_t rho, last_rho, alpha, omega;
    vector<v_t> p, v, x(result);

    vector<v_t> r(b-prod(A,x));
    vector<v_t> r_tilde(r);
    real_t initial_residual = norm_2(r);

    // silence "used uninitialized" warnings
    last_rho = 0;
    alpha = 0;

    while (iterations < max_iterations)
    {
      rho = inner_prod(conj(r_tilde), r);
      if (helpers::absolute_value(rho) == 0)
	throw std::runtime_error("bicgstab failed, rho == 0");
      if (iterations == 0)
	{
	  p = r;
	}
      else
	{
	  v_t beta = (rho/last_rho)*(alpha/omega);
	  p = r + beta*(p-omega*v);
	}

      vector<v_t> p_hat = prod(preconditioner, p);
      v = prod(A, p_hat);
      alpha = rho/inner_prod(conj(r_tilde), v);
      vector<v_t> s = r - alpha*v;

      {
        real_t norm_s = norm_2(s);
        if (norm_s < tol * initial_residual)
          {
            x += alpha*p_hat;
            break;
          }
      }

      vector<v_t> s_hat = prod(preconditioner, s);
      vector<v_t> t = prod(A, s_hat);
      omega = inner_prod(conj(t), s)/inner_prod(conj(t), t);
      x += alpha * p_hat + omega * s_hat;
      r = s - omega * t;

      {
        real_t norm_r = norm_2(r);
        if (norm_r < tol * initial_residual)
          break;
      }

      if (helpers::absolute_value(omega) == 0)
        throw std::runtime_error("bicgstab failed, omega == 0");

      last_rho = rho;

      if (debug_level)
        {
          if (debug_level >= 2 || iterations % 10 == 0)
            std::cout << double(norm_2(r)) << std::endl;
        }
      iterations++;
    }

    result = x;

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

    bicgstab_matrix_operator(const mop_type &mat, const mop_type &precon, 
			     unsigned maxit, double tol)
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

      after.clear();
      solveBiCGSTAB(m_matrix, 
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
