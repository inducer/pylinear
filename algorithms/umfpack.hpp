#ifndef HEADER_SEEN_UMFPACK_HPP
#define HEADER_SEEN_UMFPACK_HPP




#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>
#include "matrix_operator.hpp"




namespace umfpack
{
  namespace ublas = boost::numeric::ublas;
  namespace umf = boost::numeric::bindings::umfpack;




  template <typename ValueType>
  class umfpack_matrix_operator : public algorithm_matrix_operator<ValueType>,
  boost::noncopyable
  {
    typedef
      algorithm_matrix_operator<ValueType>
      super;

  public:
    typedef
      ublas::compressed_matrix<ValueType, ublas::column_major,
                               0, ublas::unbounded_array<int> >
      matrix_type;

    typedef 
      typename super::vector_type
      vector_type;

  private:
    const matrix_type                   &m_matrix;
    umf::numeric_type<ValueType>        m_numeric;

  public:
    umfpack_matrix_operator(const matrix_type &src)
    : m_matrix(src)
    { 
      umf::factor(m_matrix, m_numeric);
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
      umf::solve(m_matrix, after, before, m_numeric);
      // FIXME: honor debug levels?
    }
  };

}




#endif
