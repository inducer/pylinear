#ifndef HEADER_SEEN_GENERIC_UBLAS_HPP
#define HEADER_SEEN_GENERIC_UBLAS_HPP




#include <boost/mpl/bool.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>




namespace generic_ublas {
  namespace ublas = boost::numeric::ublas;
  namespace mpl = boost::mpl;




  template <typename ValueType, unsigned MaxElements = 2>
  class minilist
  {
  public:
    typedef ValueType value_type;
    typedef unsigned size_type;

  private:
    ValueType m_list[MaxElements];
    unsigned m_size;

  public:
    minilist()
    : m_size(0)
    { }

    minilist(const value_type &v0)
    : m_size(1)
    { 
      m_list[0] = v1;
    }

    minilist(const value_type &v0, const value_type &v1)
    : m_size(2)
    { 
      m_list[0] = v1;
      m_list[1] = v2;
    }

    void push_back(const value_type &v)
    {
#ifndef NDEBUG
      if (m_size == MaxElements)
        throw std::runtime_error("minilist has reached max size");
#endif
      m_list[m_size++] = v;
    }

    ValueType &operator[](size_type index)
    {
      return m_list[index];
    }

    const ValueType &operator[](size_type index) const
    {
      return m_list[index];
    }
  };



  // is_vector ----------------------------------------------------------------
  template <typename UblasType>
    struct is_vector { typedef mpl::false_ type; };

  template <typename ValueType>
    struct is_vector<ublas::vector<ValueType> > { typedef mpl::true_ type; };
  template <typename WrappedVector>
    struct is_vector<ublas::vector_slice<WrappedVector> > { typedef mpl::true_ type; };




  // matrix_iterator ----------------------------------------------------------
  template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type>
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

    minilist<unsigned> index() const
    {
      return minilist<unsigned>(m_it1.index1(), m_it2.index2());
    }
  };




  template <typename MatrixType>
  class matrix_iterator<MatrixType, mpl::true_> :  public boost::iterator_adaptor<
    matrix_iterator<MatrixType>, 
    typename MatrixType::iterator,
    typename MatrixType::value_type,
    boost::forward_traversal_tag>
  {
    typedef
      boost::iterator_adaptor<
      matrix_iterator<MatrixType>, 
      typename MatrixType::iterator,
      typename MatrixType::value_type,
      boost::forward_traversal_tag>
        super;

  public:
    matrix_iterator() { }

    matrix_iterator(const typename MatrixType::iterator &it)
    : super(it)
    { }

    minilist<unsigned> index() const
    {
      return minilist<unsigned>(this->base_reference().index());
    }

  private:
    friend class boost::iterator_core_access;
  };




  template <typename AE>
  matrix_iterator<AE>
  begin(ublas::matrix_expression<AE> &mat)
  {
    return matrix_iterator<AE>
      (mat().begin1(), mat().begin1().begin());
  }

  template <typename AE>
  matrix_iterator<AE>
  end(ublas::matrix_expression<AE> &mat)
  {
    return matrix_iterator<AE>
      (mat().end1(), mat().end1().end());
  }

  template <typename AE>
  matrix_iterator<AE>
  begin(ublas::vector_expression<AE> &mat)
  {
    return matrix_iterator<AE>(mat().begin());
  }

  template <typename AE>
  matrix_iterator<AE>
  end(ublas::vector_expression<AE> &mat)
  {
    return matrix_iterator<AE>(mat().end());
  }




  // shapes and subscripting --------------------------------------------------
  template <typename AE>
  minilist<unsigned> getShape(const ublas::matrix_expression<AE> &mat)
  {
    return minilist<unsigned>(mat.size1(), mat.size2());
  }

  template <typename AE>
  minilist<unsigned> getShape(const ublas::vector_expression<AE> &mat)
  {
    return minilist<unsigned>(mat.size());
  }

  template <typename AE>
  void setShape(ublas::matrix_expression<AE> &mat,
      const minilist<unsigned> &shape)
  {
    mat.resize(shape[0], shape[1]);
  }

  template <typename AE>
  void setShape(ublas::vector_expression<AE> &mat,
      const minilist<unsigned> &shape)
  {
    mat.resize(shape[0]);
  }

  template <typename AE>
  AE *newWithShape(const ublas::matrix_expression<AE> &,
      const minilist<unsigned> &shape)
  {
    return new AE(shape[0], shape[1]);
  }

  template <typename AE>
  AE *newWithShape(const ublas::vector_expression<AE> &,
      const minilist<unsigned> &shape)
  {
    return new AE(shape[0]);
  }

  template <typename AE>
  void insert(const ublas::vector_expression<AE> &m,
      const minilist<unsigned> &index,
      const typename AE::value_type &value
      )
  {
    m.insert(index[0], value);
  }

  template <typename AE>
  void insert(const ublas::matrix_expression<AE> &m,
      const minilist<unsigned> &index,
      const typename AE::value_type &value
      )
  {
    m.insert(index[0], index[1], value);
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
