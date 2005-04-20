#ifndef HEADER_SEEN_MATRIX_OPERATOR_HPP
#define HEADER_SEEN_MATRIX_OPERATOR_HPP




#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>




template <typename ValueType>
class matrix_operator
{
  public:
    // types 
    typedef unsigned size_type;
    typedef ValueType value_type;
    typedef boost::numeric::ublas::vector<ValueType> vector_type;

    // interface
    virtual ~matrix_operator() { }

    virtual unsigned size1() const = 0;
    virtual unsigned size2() const = 0;

    /** Before using apply, before and after must have the correct size.
     */
    virtual void apply(const vector_type &before, vector_type &after) const
    {
      if (size2() != before.size() || size1() != after.size())
	throw std::runtime_error("invalid vector sizes in matrix_operator::apply");
    }

    // matrix_expression compatibility
    const matrix_operator &operator()() const
    {
      return *this;
    }
};




template <typename ValueType>
class algorithm_matrix_operator : public matrix_operator<ValueType>
{
  protected:
    unsigned m_lastIterationCount;
    unsigned m_debugLevel;

  public:
    algorithm_matrix_operator()
      : m_lastIterationCount(0), m_debugLevel(0)
      { }

    unsigned getDebugLevel() const
    {
      return m_debugLevel;
    }

    void setDebugLevel(unsigned dl)
    {
      m_debugLevel = dl;
    }

    unsigned getLastIterationCount() const 
    {
      return m_lastIterationCount;
    }
};




template <typename ValueType>
class iterative_solver_matrix_operator : public algorithm_matrix_operator<ValueType>
{
  protected:
    unsigned m_maxIterations;
    double m_tolerance;

  public:
    iterative_solver_matrix_operator(unsigned maxit = 0, double tol = 0)
      : m_maxIterations(maxit), m_tolerance(tol)
      { }

    unsigned getMaxIterations() const
    {
      return m_maxIterations;
    }

    void setMaxIterations(unsigned maxit)
    {
      m_maxIterations = maxit;
    }

    double getTolerance() const
    {
      return m_tolerance;
    }

    void setTolerance(double tol)
    {
      m_tolerance = tol;
    }
};




template <typename MatrixType>
class ublas_matrix_operator : public matrix_operator<typename MatrixType::value_type>
{
    const MatrixType &m_matrix;
    typedef matrix_operator<typename MatrixType::value_type> super;

  public:
    typedef 
      typename matrix_operator<typename MatrixType::value_type>::vector_type
      vector_type;
    
    ublas_matrix_operator(const MatrixType &m)
    : m_matrix(m)
    { 
    }
    
    unsigned size1() const
    {
      return m_matrix.size1();
    }

    unsigned size2() const
    {
      return m_matrix.size2();
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      super::apply(before, after);
      
      using namespace boost::numeric::ublas;
      //after = prod(m_matrix, before);
      axpy_prod(m_matrix, before, after, /*init*/ true);
    }
};




template <typename ValueType>
class identity_matrix_operator : public matrix_operator<ValueType>
{
    typedef
      matrix_operator<ValueType>
      super;

    unsigned m_size;

  public:
    typedef 
      typename super::vector_type
      vector_type;
    
    identity_matrix_operator(unsigned size)
      : m_size(size)
      { }

    unsigned size2() const
    {
      return m_size;
    }
    unsigned size1() const
    {
      return m_size;
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      super::apply(before, after);
      after = before;
    }
};




template <typename ValueType>
class composite_matrix_operator : public matrix_operator<ValueType>
{
    typedef
      matrix_operator<ValueType>
      super;

    const super       &m_outer, &m_inner;

  public:
    typedef 
      typename super::vector_type
      vector_type;
    
    composite_matrix_operator(const super &outer, const super &inner)
    : m_outer(outer), m_inner(inner)
    { 
      if (m_inner.size1() != m_outer.size2())
        throw std::runtime_error("composite_matrix_operator: sizes do not match");
    }

    unsigned size1() const
    {
      return m_outer.size1();
    }

    unsigned size2() const
    {
      return m_inner.size2();
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      super::apply(before, after);

      vector_type temp(m_inner.size1());
      m_inner.apply(before, temp);
      m_outer.apply(temp, after);
    }
};




template <typename ValueType>
class sum_of_matrix_operators : public matrix_operator<ValueType>
{
    typedef
      matrix_operator<ValueType>
      super;

    const super       &m_op1, &m_op2;

  public:
    typedef 
      typename super::vector_type
      vector_type;
    
    sum_of_matrix_operators (const super &op1, const super &op2)
    : m_op1(op1), m_op2(op2)
    { 
      if (m_op1.size1() != m_op2.size1())
        throw std::runtime_error("sum_of_matrix_operators: sizes do not match");
      if (m_op1.size2() != m_op2.size2())
        throw std::runtime_error("sum_of_matrix_operators: sizes do not match");
    }

    unsigned size1() const
    {
      return m_op1.size1();
    }

    unsigned size2() const
    {
      return m_op1.size2();
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      super::apply(before, after);

      vector_type temp(size1());
      m_op1.apply(before, temp);
      m_op2.apply(before, after);

      after += temp;
    }
};




template <typename ValueType>
class complex_matrix_operator_adaptor : public matrix_operator<std::complex<ValueType> >
{
    typedef
      matrix_operator<std::complex<ValueType> >
      super;
    typedef
      matrix_operator<ValueType>
      real_op;
    typedef 
      typename real_op::vector_type 
      real_vector_type;

    const real_op       &m_real, &m_imaginary;

  public:
    typedef 
      typename super::vector_type
      vector_type;
    
    complex_matrix_operator_adaptor(const real_op &real_part, const real_op &imaginary_part)
    : m_real(real_part), m_imaginary(imaginary_part)
    { 
      if (m_real.size1() != m_imaginary.size1())
        throw std::runtime_error("complex_matrix_operator_adaptor: sizes do not match");
      if (m_real.size2() != m_imaginary.size2())
        throw std::runtime_error("complex_matrix_operator_adaptor: sizes do not match");
    }

    unsigned size1() const
    {
      return m_real.size1();
    }

    unsigned size2() const
    {
      return m_real.size2();
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      super::apply(before, after);

      real_vector_type before_real(real(before)), before_imag(imag(before));
      real_vector_type after_real(real(after)), after_imag(imag(after));
      real_vector_type after_real_2(real(after)), after_imag_2(imag(after));

      m_real.apply(before_real, after_real);
      m_imaginary.apply(before_imag, after_real_2);
      after_real_2 *= -1;

      m_imaginary.apply(before_real, after_imag);
      m_real.apply(before_imag, after_imag_2);

      after = after_real + after_real_2 + 
	std::complex<ValueType>(0,1) * (after_imag + after_imag_2);
    }
};




template <typename ValueType>
class scalar_multiplication_matrix_operator : public matrix_operator<ValueType>
{
    typedef
      matrix_operator<ValueType>
      super;

    ValueType m_factor;
    unsigned m_size;

  public:
    typedef 
      typename super::vector_type
      vector_type;
    
    scalar_multiplication_matrix_operator(const ValueType &factor, unsigned size)
    : m_factor(factor), m_size(size)
    { 
    }

    unsigned size1() const
    {
      return m_size;
    }

    unsigned size2() const
    {
      return m_size;
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      super::apply(before, after);
      after = m_factor * before;
    }
};




// generic prod() interface ---------------------------------------------------
template <typename ValueType>
boost::numeric::ublas::vector<ValueType> prod(
    const matrix_operator<ValueType> &mo,
    const boost::numeric::ublas::vector<ValueType> &vec)
{
  boost::numeric::ublas::vector<ValueType> result(mo.size1());
  mo.apply(vec, result);
  return result;
}




#endif




// EMACS-FORMAT-TAG
//
// Local Variables:
// mode: C++
// eval: (c-set-style "stroustrup")
// eval: (c-set-offset 'access-label -2)
// eval: (c-set-offset 'inclass '++)
// c-basic-offset: 2
// tab-width: 8
// End:
