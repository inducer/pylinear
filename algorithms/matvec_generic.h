#ifndef HEADER_SEEN_GENERIC_ITERATOR_H
#define HEADER_SEEN_GENERIC_ITERATOR_H




#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>



namespace generic_iterator {
  namespace ublas = boost::numeric::ublas;




  template <typename MatrixType>
  class matrix_iterator :  public boost::iterator_facade<
    matrix_iterator<MatrixType>, 
    typename MatrixType::value_type,
    boost::forward_traversal_tag>
  {
    typedef typename MatrixType::iterator1 it1_t;
    typedef typename MatrixType::iterator2 it2_t;

    it1_t       m_it1;
    it2_t       m_it2;

  public:
    matrix_iterator() { }

    matrix_iterator(const it1_t &it1, const it2_t &it2)
    : m_it1(it1), m_it2(it2)
    { }

  private:
    friend class boost::iterator_core_access;

    void increment() 
    {
      m_it2++;
      if (m_it2 == m_it1.end())
      {
        m_it1++;
        m_it2 = m_it1.begin();
      }
    }

    bool equal(matrix_iterator const& other) const
    {
      return m_it1 == other.m_it1 && m_it2 == other.m_it2;
    }

    typename MatrixType::value_type &dereference() const 
    {
      return *m_it2; 
    }
  };




  template <typename AE>
  class matrix_iterator<ublas::vector_expression<AE> > :  public boost::iterator_adaptor<
    matrix_iterator<ublas::vector_expression<AE> >, 
    typename ublas::vector_expression<AE>::value_type,
    boost::forward_traversal_tag>
  {
  public:
    matrix_iterator() { }

    /*
    template <class OtherValue>
    matrix_iterator(matrix_iterator<OtherValue> const& other, 
        typename boost::enable_if<
        boost::is_convertible<OtherValue*,Value*>, enabler >::type = enabler())
    : super_t(other.base()) { }
    */

  private:
    friend class boost::iterator_core_access;
  };




  template <typename AE>
  matrix_iterator<ublas::matrix_expression<AE> >
  begin(ublas::matrix_expression<AE> &mat)
  {
    return matrix_iterator<ublas::matrix_expression<AE> >
      (mat.begin1(), mat.begin1().begin());
  }

  template <typename AE>
  matrix_iterator<ublas::matrix_expression<AE> >
  end(ublas::matrix_expression<AE> &mat)
  {
    return matrix_iterator<ublas::matrix_expression<AE> >
      (mat.end1(), mat.end1().end());
  }

  template <typename AE>
  matrix_iterator<ublas::vector_expression<AE> >
  begin(ublas::matrix_expression<AE> &mat)
  {
    return matrix_iterator<ublas::vector_expression<AE> >(mat.begin());
  }

  template <typename AE>
  matrix_iterator<ublas::vector_expression<AE> >
  end(ublas::matrix_expression<AE> &mat)
  {
    return matrix_iterator<ublas::vector_expression<AE> >(mat.end());
  }





  /*
  template <typename MatrixType>
  class full_matrix_iterator :  public boost::iterator_facade<
    matrix_iterator, 
    typename MatrixType::value_type,
    boost::forward_traversal_tag>
  {
    MatrixType *m_matrix;
    unsigned m_index1,m_index2;

  public:
    matrix_iterator() { }

    matrix_iterator(MatrixType &mat, unsigned i1, unsigned i2)
    : m_matrix(&mat), m_index1(i1), m_index2(i2)
    { }

  private:
    friend class boost::iterator_core_access;

    void increment() 
    {
      m_index2++;
      if (m_index2 == m_i1.end())
      {
        m_it1++;
        m_it2 = m_it1.begin();
      }
    }

    bool equal(matrix_iterator const& other) const
    {
      return m_matrix == other.m_matrix && m_index1 == other.m_index1 &&
        m_index2 == other.m_index2;
    }

    node_base& dereference() const 
    {
      return (*m_matrix)(m_index1, m_index2); 
    }
  };
  */
}




#endif
