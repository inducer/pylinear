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

  namespace detail
  {
    class begin_tag { };
    class end_tag { };
  }




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
      m_list[0] = v0;
    }

    minilist(const value_type &v0, const value_type &v1)
    : m_size(2)
    { 
      m_list[0] = v0;
      m_list[1] = v1;
    }

    size_type size() const
    {
      return m_size;
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
  template <typename WrappedVector>
    struct is_vector<ublas::matrix_row<WrappedVector> > { typedef mpl::true_ type; };
  template <typename WrappedVector>
    struct is_vector<ublas::matrix_column<WrappedVector> > { typedef mpl::true_ type; };




  // matrix_iterator ----------------------------------------------------------
  template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type>
  class matrix_iterator :  public boost::iterator_facade<
    matrix_iterator<MatrixType>,  // Base
    typename MatrixType::value_type, // Value
    boost::forward_traversal_tag, // CategoryOrTraversal
    typename MatrixType::iterator2::reference> // Reference
  {
    typedef typename MatrixType::iterator1 it1_t;
    typedef typename MatrixType::iterator2 it2_t;

    it1_t       m_it1;
    it2_t       m_it2;

  public:
    matrix_iterator() { }

    matrix_iterator(MatrixType &mat, detail::begin_tag)
    : m_it1(mat.begin1()), m_it2(m_it1.begin())
    { 
      validate();
    }

    matrix_iterator(MatrixType &mat, detail::end_tag)
    : m_it1(mat.end1()), m_it2(m_it1.begin())
    { }

    minilist<unsigned> index() const
    {
      return minilist<unsigned>(m_it2.index1(), m_it2.index2());
    }

  private:
    friend class boost::iterator_core_access;

    void validate()
    {
      // this makes sure that the iterator points to an existing element
      while (m_it1 != m_it1().end1() && m_it2 == m_it1.end())
      {
        m_it1++;
        m_it2 = m_it1.begin();
      }
    }

    void increment() 
    {
      m_it2++;
      validate();
    }

    bool equal(matrix_iterator const& other) const
    {
      return m_it1 == other.m_it1 && m_it2 == other.m_it2;
    }

    typename MatrixType::iterator2::reference dereference() const 
    {
      return *m_it2; 
    }
  };




  template <typename MatrixType>
  class matrix_iterator<MatrixType, mpl::true_> :  public boost::iterator_adaptor<
    matrix_iterator<MatrixType>, 
    typename MatrixType::iterator,
    typename MatrixType::value_type,
    boost::forward_traversal_tag,
    typename MatrixType::iterator::reference>
  {
    typedef
      boost::iterator_adaptor<
      matrix_iterator<MatrixType>, 
      typename MatrixType::iterator,
      typename MatrixType::value_type,
      boost::forward_traversal_tag,
      typename MatrixType::iterator::reference>
        super;

  public:
    matrix_iterator() { }

    matrix_iterator(MatrixType &mat, detail::begin_tag)
    : super(mat.begin())
    { }

    matrix_iterator(MatrixType &mat, detail::end_tag)
    : super(mat.end())
    { }

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




  template <typename MatrixType>
  matrix_iterator<MatrixType> begin(MatrixType &mat)
  {
    return matrix_iterator<MatrixType>(mat, detail::begin_tag());
  }

  template <typename MatrixType>
  matrix_iterator<MatrixType> end(MatrixType &mat)
  {
    return matrix_iterator<MatrixType>(mat, detail::end_tag());
  }




  // shapes and subscripting --------------------------------------------------
  namespace detail
  {
    template <typename MatrixType>
    inline minilist<unsigned> getShape(const MatrixType &mat, mpl::false_)
    {
      return minilist<unsigned>(mat.size1(), mat.size2());
    }

    template <typename MatrixType>
    inline minilist<unsigned> getShape(const MatrixType &mat, mpl::true_)
    {
      return minilist<unsigned>(mat.size());
    }
  }

  template <typename MatrixType>
  inline minilist<unsigned> getShape(const MatrixType &mat)
  {
    return detail::getShape(mat, typename is_vector<MatrixType>::type());
  }




  namespace detail
  {
    template <typename MatrixType>
    inline void setShape(MatrixType &mat, const minilist<unsigned> &shape, mpl::false_)
    {
      mat.resize(shape[0], shape[1]);
    }

    template <typename MatrixType>
    inline void setShape(MatrixType &mat, const minilist<unsigned> &shape, mpl::true_)
    {
      mat.resize(shape[0]);
    }
  }

  template <typename MatrixType>
  inline void setShape(MatrixType &mat, const minilist<unsigned> &shape)
  {
    detail::setShape(mat, shape, typename is_vector<MatrixType>::type());
  }






  namespace detail
  {
    template <typename MatrixType>
    inline MatrixType *newWithShape(const minilist<unsigned> &shape, mpl::false_)
    {
      return new MatrixType(shape[0], shape[1]);
    }

    template <typename MatrixType>
    inline MatrixType *newWithShape(const minilist<unsigned> &shape, mpl::true_)
    {
      return new MatrixType(shape[0]);
    }
  }



  template <typename MatrixType>
  MatrixType *newWithShape(const minilist<unsigned> &shape)
  {
    return detail::newWithShape<MatrixType>(shape, typename is_vector<MatrixType>::type());
  }




  namespace detail
  {
    template <typename MatrixType>
    inline void insert_element(
        MatrixType &mat,
        const minilist<unsigned> &index, 
        const typename MatrixType::value_type &value,
        mpl::false_)
    {
      mat.insert_element(index[0], index[1], value);
    }

    template <typename MatrixType>
    inline void insert_element(
        MatrixType &mat,
        const minilist<unsigned> &index, 
        const typename MatrixType::value_type &value,
        mpl::true_)
    {
      mat.insert_element(index[0], value);
    }
  }

  template <typename MatrixType>
  inline void insert_element(
      MatrixType &mat,
      const minilist<unsigned> &index, 
      const typename MatrixType::value_type &value)
  {
    detail::insert_element(mat,index, value, typename is_vector<MatrixType>::type());
  }




  namespace detail
  {
    template <typename MatrixType>
    inline void set(
        MatrixType &mat,
        const minilist<unsigned> &index, 
        const typename MatrixType::value_type &value,
        mpl::false_)
    {
      mat(index[0], index[1]) = value;
    }

    template <typename MatrixType>
    inline void set(
        MatrixType &mat,
        const minilist<unsigned> &index, 
        const typename MatrixType::value_type &value,
        mpl::true_)
    {
      mat[index[0]] = value;
    }
  }

  template <typename MatrixType>
  inline void set(
      MatrixType &mat,
      const minilist<unsigned> &index, 
      const typename MatrixType::value_type &value)
  {
    detail::set(mat,index, value, typename is_vector<MatrixType>::type());
  }
}




#endif
