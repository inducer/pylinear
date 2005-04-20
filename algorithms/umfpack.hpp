#ifndef HEADER_SEEN_UMFPACK_HPP
#define HEADER_SEEN_UMFPACK_HPP




#include <stdexcept>
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
      process_umfpack_error(umf::factor(m_matrix, m_numeric));
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
      process_umfpack_error(umf::solve(m_matrix, after, before, m_numeric));
      // FIXME: honor debug levels?
    }
  private:
    static void process_umfpack_error(int umf_error) 
    {
      switch (umf_error)
      {
      case UMFPACK_OK: 
        return;
      case UMFPACK_ERROR_out_of_memory: 
        throw std::runtime_error("umfpack: out of memory");
      case UMFPACK_ERROR_invalid_Numeric_object: 
        throw std::runtime_error("umfpack: invalid numeric object");
      case UMFPACK_ERROR_invalid_Symbolic_object:
        throw std::runtime_error("umfpack: invalid symbolic object");
      case UMFPACK_ERROR_argument_missing:
        throw std::runtime_error("umfpack: argument missing");
      case UMFPACK_ERROR_n_nonpositive:
        throw std::runtime_error("umfpack: n non-positive");
      case UMFPACK_ERROR_invalid_matrix:
        throw std::runtime_error("umfpack: invalid matrix");
      case UMFPACK_ERROR_different_pattern:
        throw std::runtime_error("umfpack: different pattern");
      case UMFPACK_ERROR_invalid_system:
        throw std::runtime_error("umfpack: invalid system");
      case UMFPACK_ERROR_invalid_permutation:
        throw std::runtime_error("umfpack: invalid permutation");
      case UMFPACK_ERROR_internal_error:
        throw std::runtime_error("umfpack: internal error");
      case UMFPACK_ERROR_file_IO:
        throw std::runtime_error("umfpack: file i/o error");
      default:
        throw std::runtime_error("umfpack: invalid error code");
      }
    }
  };

}




#endif
