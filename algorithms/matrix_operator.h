#ifndef HEADER_SEEN_MATRIX_OPERATOR_H
#define HEADER_SEEN_MATRIX_OPERATOR_H




#include <boost/numeric/ublas/vector.hpp>




template <typename ValueType>
class matrix_operator
{
  public:
    typedef unsigned size_type;
    typedef ValueType value_type;

    typedef boost::numeric::ublas::vector<ValueType> vector_type;

    virtual unsigned size1() const = 0;
    virtual unsigned size2() const = 0;
    virtual void apply(const vector_type &before, vector_type &after) const = 0;

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

  public:
    typedef 
      typename matrix_operator<typename MatrixType::value_type>::vector_type
      vector_type;
    
    ublas_matrix_operator(const MatrixType &m)
    : m_matrix(m)
    { }
    
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
      after = prod(m_matrix, before);
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
      after = before;
    }
};




template <typename ValueType>
boost::numeric::ublas::vector<ValueType> prod(
    const matrix_operator<ValueType> &mo,
    const boost::numeric::ublas::vector<ValueType> &vec)
{
  boost::numeric::ublas::vector<ValueType> result;
  mo.apply(vec, result);
  return result;
}




#endif
