#ifndef HEADER_SEEN_UMFPACK_H
#define HEADER_SEEN_UMFPACK_H




#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>
#include <matrix_operator.h>




namespace umfpack
{
  namespace ublas = boost::numeric::ublas;
  namespace umf = boost::numeric::bindings::umfpack;




  template <typename ValueType>
  class umfpack_matrix_operator : public algorithm_matrix_operator<ValueType>,
  boost::noncopyable
  {
    ublas::compressed_matrix<ValueType, ublas::column_major> m_matrix;
    umf::numeric_type<ValueType>        m_numeric;

    typedef
      algorithm_matrix_operator<ValueType>
      super;
  public:
    typedef 
      typename super::vector_type
      vector_type;
    
    template <typename MatrixType>
    umfpack_matrix_operator(const MatrixType &src)
    : m_matrix(src)
    { 
      umf::factor(m_matrix, m_numeric);
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
      umf::solve(m_matrix, after, before, m_numeric);
      // FIXME: honor debug levels?
    }
  };

}
#endif
