#include <complex>
#include <string>
#include <cmath>

#include <boost/python.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/utility/base_from_member.hpp>




using namespace boost;
namespace ublas = boost::numeric::ublas;
using boost::python::class_;
using boost::python::enum_;
using boost::python::self;
using boost::python::def;




// typecode support -----------------------------------------------------------
enum SupportedElementTypes {
  Float64,
  Complex64,
};




inline SupportedElementTypes getTypeCode(double) { return Float64; }
inline SupportedElementTypes getTypeCode(std::complex<double>) { return Complex64; }




// managed adaptors -----------------------------------------------------------
namespace
{
  template <typename AdaptedMatrix>
  class managed_symmetric_adaptor : 
    private base_from_member<std::auto_ptr<AdaptedMatrix> >,
    public ublas::symmetric_adaptor<AdaptedMatrix>
  {
    typedef ublas::symmetric_adaptor<AdaptedMatrix> super;
    typedef boost::base_from_member<std::auto_ptr<AdaptedMatrix> > m_data;
  public:
    typedef AdaptedMatrix adapted_matrix;
    typedef typename super::size_type size_type;
    typedef typename super::value_type value_type;

    managed_symmetric_adaptor()
      : m_data(new AdaptedMatrix()), super(*m_data::member)
      {
      }

    managed_symmetric_adaptor(size_type s1, size_type s2)
      : m_data(new AdaptedMatrix(s1, s1)), super(*m_data::member)
      {
        if (s1 != s2)
          throw std::runtime_error( "symmetric matrices are quadratic" );
      }

    managed_symmetric_adaptor(const managed_symmetric_adaptor &m)
      : m_data(new AdaptedMatrix(*m.member)), super(*m_data::member)
      {
      }

    template <class AE>
    managed_symmetric_adaptor(const ublas::matrix_expression<AE> &ae)
      : m_data(new AdaptedMatrix(ae().size1(), ae().size1())), super(*m_data::member)
      {
        if (ae().size1() != ae().size2())
          throw std::runtime_error( "symmetric matrices are quadratic" );
        super::operator=(ae);
      }

    void resize(size_type s1, size_type s2)
    {
      if (s1 != s2)
        throw std::runtime_error( "symmetric matrices are quadratic" );
      m_data::member->resize(s1, s2);
    }

    void insert(size_type i1, size_type i2, const value_type &val)
    {
      super::operator()(i1, i2) = val;
    }
    
    managed_symmetric_adaptor &operator=(const managed_symmetric_adaptor &m)
    {
      *m_data::member.operator=(*m.member);
    }

    template <class AE>
    managed_symmetric_adaptor &operator=(const ublas::matrix_expression<AE> &ae)
    {
      if (ae().size1() != ae().size2())
        throw std::runtime_error( "symmetric matrices are quadratic" );
      m_data::member->resize(ae.size1(), ae.size2());
      super::operator=(ae);
    }
  };

  template <typename AdaptedMatrix>
  class managed_hermitian_adaptor : 
    private base_from_member<std::auto_ptr<AdaptedMatrix> >,
    public ublas::hermitian_adaptor<AdaptedMatrix>
  {
    typedef ublas::hermitian_adaptor<AdaptedMatrix> super;
    typedef boost::base_from_member<std::auto_ptr<AdaptedMatrix> > m_data;
  public:
    typedef AdaptedMatrix adapted_matrix;
    typedef typename super::size_type size_type;
    typedef typename super::value_type value_type;

    managed_hermitian_adaptor()
      : m_data(new AdaptedMatrix()), super(*m_data::member)
      {
      }

    managed_hermitian_adaptor(size_type s1, size_type s2)
      : m_data(new AdaptedMatrix(s1, s1)), super(*m_data::member)
      {
        if (s1 != s2)
          throw std::runtime_error( "hermitian matrices are quadratic" );
      }

    managed_hermitian_adaptor(const managed_hermitian_adaptor &m)
      : m_data(new AdaptedMatrix(*m.member)), super(*m_data::member)
      {
      }

    template <class AE>
    managed_hermitian_adaptor(const ublas::matrix_expression<AE> &ae)
      : m_data(new AdaptedMatrix(ae().size1(), ae().size1())), super(*m_data::member)
      {
        if (ae().size1() != ae().size2())
          throw std::runtime_error( "hermitian matrices are quadratic" );
        super::operator=(ae);
      }

    void resize(size_type s1, size_type s2)
    {
      if (s1 != s2)
        throw std::runtime_error( "hermitian matrices are quadratic" );
      m_data::member->resize(s1, s2);
    }

    void insert(size_type i1, size_type i2, const value_type &val)
    {
      super::operator()(i1, i2) = val;
    }
    
    managed_hermitian_adaptor &operator=(const managed_hermitian_adaptor &m)
    {
      *m_data::member.operator=(*m.member);
    }

    template <class AE>
    managed_hermitian_adaptor &operator=(const ublas::matrix_expression<AE> &ae)
    {
      if (ae().size1() != ae().size2())
        throw std::runtime_error( "hermitian matrices are quadratic" );
      m_data::member->resize(ae.size1(), ae.size2());
      super::operator=(ae);
    }
  };
}




// metaprogramming helpers ----------------------------------------------------
namespace value_type_promotion
{
  using std::complex;

  template <typename A, typename B>
  struct bigger_type
  {
  };

  template <typename A>
  struct bigger_type<A,A> { typedef A type; };

  template <> struct bigger_type<float, double> { typedef double type; };
  template <> struct bigger_type<float, complex<float> > { typedef complex<float> type; };
  template <> struct bigger_type<float, complex<double> > { typedef complex<double> type; };

  template <> struct bigger_type<double, complex<float> > { typedef complex<double> type; };
  template <> struct bigger_type<double, complex<double> > { typedef complex<double> type; };

  template <> struct bigger_type<complex<float>, complex<double> > { typedef complex<double> type; };

  // and the other way around
  template <> struct bigger_type<double, float> { typedef double type; };
  template <> struct bigger_type<complex<float>, float> { typedef complex<float> type; };
  template <> struct bigger_type<complex<double>, float> { typedef complex<double> type; };

  template <> struct bigger_type<complex<float>, double> { typedef complex<double> type; };
  template <> struct bigger_type<complex<double>, double> { typedef complex<double> type; };

  template <> struct bigger_type<complex<double>, complex<float> > { typedef complex<double> type; };
}




// is_vector
template <typename UblasType>
struct is_vector { typedef mpl::false_ type; };

template <typename ValueType>
struct is_vector<ublas::vector<ValueType> > { typedef mpl::true_ type; };
template <typename WrappedVector>
struct is_vector<ublas::vector_slice<WrappedVector> > { typedef mpl::true_ type; };





// decomplexify
template <typename T>
struct decomplexify
{
  typedef T type;
};

template <typename ELT>
struct decomplexify<std::complex<ELT> >
{
  typedef ELT type;
};




// get_computation_result_type
template <typename MatrixType>
struct get_computation_result_type 
{ typedef MatrixType type; };




// change_value_type
template <typename MatrixType, typename NewValueType>
struct change_value_type { };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::matrix<OldValueType>, NewValueType>
{ typedef ublas::matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::sparse_matrix<OldValueType>, NewValueType>
{ typedef ublas::sparse_matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::coordinate_matrix<OldValueType>, NewValueType>
{ typedef ublas::coordinate_matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::compressed_matrix<OldValueType>, NewValueType>
{ typedef ublas::compressed_matrix<NewValueType> type; };

template <typename OldValueType, typename NewValueType>
struct change_value_type<ublas::vector<OldValueType>, NewValueType>
{ typedef ublas::vector<NewValueType> type; };

template <typename ContainedMatrixType, typename NewValueType>
struct change_value_type<managed_symmetric_adaptor<ContainedMatrixType>, NewValueType>
{ 
  typedef 
    managed_symmetric_adaptor<
    typename change_value_type<ContainedMatrixType, NewValueType>::type> type; 
};

template <typename ContainedMatrixType, typename NewValueType>
struct change_value_type<managed_hermitian_adaptor<ContainedMatrixType>, NewValueType>
{ 
  typedef 
    managed_hermitian_adaptor<
    typename change_value_type<ContainedMatrixType, NewValueType>::type> type; 
};




// get_corresponding_vector_type
template <typename MatrixType>
struct get_corresponding_vector_type 
{ typedef ublas::vector<typename MatrixType::value_type> type; };




namespace {
// universal functions --------------------------------------------------------
template <typename MatrixType>
inline SupportedElementTypes typecode(const MatrixType &)
{ 
  return getTypeCode(typename MatrixType::value_type());
}




template <typename MatrixType>
inline MatrixType *copyNew(MatrixType &m)
{
  return new MatrixType(m);
}




template <typename Op1, typename Op2>
struct prodMatMatWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_value_type;
  typedef 
    typename ublas::matrix<result_value_type> 
    result_type;

  inline static result_type *apply(Op1 &op1, Op2 &op2)
  {
    return new result_type(prod(op1, op2));
  }
};




template <typename Op1, typename Op2>
struct prodMatVecWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_value_type;
  typedef 
    typename ublas::vector<result_value_type> 
    result_type;

  inline static result_type *apply(Op1 &op1, Op2 &op2)
  {
    return new result_type(prod(op1, op2));
  }
};




template <typename Op1, typename Op2>
struct inner_prodWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_type;

  inline static result_type apply(Op1 &op1, Op2 &op2)
  {
    return inner_prod(op1, op2);
  }
};




template <typename Op1, typename Op2>
struct outer_prodWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_value_type;
  typedef 
    typename ublas::matrix<result_value_type> 
    result_type;

  inline static result_type *apply(const Op1 &op1, const Op2 &op2)
  {
    return new result_type(outer_prod(op1, op2));
  }
};




template <typename MatrixType>
inline typename get_computation_result_type<MatrixType>::type *
transposeMatrix(const MatrixType &m)
{
  return new
    typename get_computation_result_type<MatrixType>::type
    (trans(m));
}




template <typename MatrixType>
inline typename get_computation_result_type<MatrixType>::type *
hermiteMatrix(const MatrixType &m)
{
  return new
    typename get_computation_result_type<MatrixType>::type
    (herm(m));
}




template <typename MatrixType>
struct realWrapper
{
  typedef 
    typename change_value_type<
      typename get_computation_result_type<MatrixType>::type, 
      typename decomplexify<typename MatrixType::value_type>::type>::type
    result_type;

  inline static result_type *apply(MatrixType &m)
  {
    return new result_type(real(m));
  }
};




template <typename MatrixType>
struct imagWrapper
{
  typedef 
    typename change_value_type<
      typename get_computation_result_type<MatrixType>::type, 
      typename decomplexify<typename MatrixType::value_type>::type>::type
    result_type;

  inline static result_type *apply(MatrixType &m)
  {
    return new result_type(imag(m));
  }
};




template <typename MatrixType>
struct conjugateWrapper
{
  typedef
    typename get_computation_result_type<MatrixType>::type
    result_type;

  inline static result_type *apply(const MatrixType &m)
  {
    return new result_type(conj(m));
  }
};




template <typename T>
inline std::string stringify(const T &obj)
{
  std::stringstream stream;
  stream << obj;
  return stream.rdbuf()->str();
}




namespace ufuncs
{




#define MAKE_UNARY_FUNCTION_ADAPTER(f) \
  template <typename T> \
  struct SimpleFunctionAdapter_##f \
  { \
    inline T operator()(const T &x) \
    { \
      return f(x); \
    } \
  }

  // FIXME complex to real
  /*
  MAKE_UNARY_FUNCTION_ADAPTER(absolute_value);
  MAKE_UNARY_FUNCTION_ADAPTER(arg);
  */

  // every type
  MAKE_UNARY_FUNCTION_ADAPTER(cos);
  MAKE_UNARY_FUNCTION_ADAPTER(cosh);
  MAKE_UNARY_FUNCTION_ADAPTER(exp);
  MAKE_UNARY_FUNCTION_ADAPTER(log);
  MAKE_UNARY_FUNCTION_ADAPTER(log10);
  MAKE_UNARY_FUNCTION_ADAPTER(sin);
  MAKE_UNARY_FUNCTION_ADAPTER(sinh);
  MAKE_UNARY_FUNCTION_ADAPTER(sqrt);
  MAKE_UNARY_FUNCTION_ADAPTER(tan);
  MAKE_UNARY_FUNCTION_ADAPTER(tanh);
#undef MAKE_UNARY_FUNCTION_ADAPTER




  template <typename MatrixType, typename Function, typename _is_vector = typename is_vector<MatrixType>::type>
  struct UnaryUfuncApplicator
  {
    typedef 
      typename get_computation_result_type<MatrixType>::type 
      result_type;

    static result_type *apply(const MatrixType &m)
    {
      Function f;

      std::auto_ptr<result_type> new_mat(new result_type(m.size1(), m.size2()));

      typedef typename MatrixType::const_iterator1 it1_t;
      typedef typename MatrixType::const_iterator2 it2_t;
      for (it1_t it1 = m.begin1(); it1 != m.end1(); ++it1) 
        for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
          new_mat->insert(it2.index1(), it2.index2(), f(*it2));

      return new_mat.release();
    }
  };




  template <typename MatrixType, typename Function>
  struct UnaryUfuncApplicator<MatrixType, Function, mpl::true_>
  {
    typedef 
      typename get_computation_result_type<MatrixType>::type 
      result_type;
    static result_type *apply(const MatrixType &m)
    {
      Function f;

      std::auto_ptr<result_type> new_mat(new result_type(m. size()));

      typedef typename MatrixType::const_iterator it_t;
      for (it_t it = m.begin(); it != m.end(); ++it) 
        new_mat->insert(it.index(), f(*it));

      return new_mat.release();
    }
  };
}




// helpers --------------------------------------------------------------------
template <typename T>
T extractOneTuple(const python::tuple &tup, T)
{
  if (tup.attr("__len__") != 1)
    throw std::runtime_error("expected tuple of length 1");
  return python::extract<T>(tup[0]);
}




template <typename T>
void extractTwoTuple(const python::tuple &tup, T &i, T &j)
{
  if (tup.attr("__len__") != 2)
    throw std::runtime_error("expected tuple of length 2");
  i = python::extract<T>(tup[0]);
  j = python::extract<T>(tup[2]);
}




// shape accessors ------------------------------------------------------------
template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct getShape
{
  static python::object apply(const MatrixType &m)
  { 
    return python::make_tuple(m.size1(), m.size2());
  }
};




template <typename MatrixType>
struct getShape<MatrixType, mpl::true_>
{
  static python::object apply(const MatrixType &m)
  { 
    return python::make_tuple(m.size());
  }
};




template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct setShape
{
  static void apply(MatrixType &m, const python::tuple &new_shape)
  { 
    typename MatrixType::size_type h,w;
    extractTwoTuple(new_shape, h, w);
    m.resize(h,w);
  }
};




template <typename MatrixType>
struct setShape<MatrixType, mpl::true_>
{
  static void apply(MatrixType &m, const python::tuple &new_shape)
  { 
    m.resize(extractOneTuple(new_shape, typename MatrixType::size_type()));
  }
};




// iterator interface ---------------------------------------------------------
template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct key_iterator_result_generator
{
  typedef python::object result_type;

  static result_type apply(typename MatrixType::iterator2 it)
  {
    return python::make_tuple(it.index1(), it.index2());
  }
};




template <typename MatrixType>
struct key_iterator_result_generator<MatrixType, mpl::true_>
{
  typedef typename MatrixType::size_type result_type;

  static result_type apply(typename MatrixType::iterator it)
  {
    return it.index();
  }
};




template <typename MatrixType>
struct value_iterator_result_generator
{
  typedef 
    typename MatrixType::value_type
    result_type;

  template <typename IteratorType>
  static result_type apply(IteratorType it)
  {
    return *it;
  }
};




template <typename MatrixType, typename ResultGenerator, typename _is_vector = typename is_vector<MatrixType>::type >
struct python_matrix_iterator
{
  typedef
    typename ResultGenerator::result_type 
    result_type;

  ResultGenerator m_generator;

  typename MatrixType::iterator1 m_iterator1;
  typename MatrixType::iterator2 m_iterator2;

  python_matrix_iterator *iter()
  {
    return this;
  }

  result_type next()
  {
    if (m_iterator2 == m_iterator1.end())
    {
      if (m_iterator1 == m_iterator1().end1())
      {
        PyErr_SetNone(PyExc_StopIteration);
        throw python::error_already_set();
      }

      m_iterator1++;
      m_iterator2 = m_iterator1.begin();

      if (m_iterator1 == m_iterator1().end1())
      {
        PyErr_SetNone(PyExc_StopIteration);
        throw python::error_already_set();
      }
    }

    result_type result = m_generator.apply(m_iterator2);
    m_iterator2++;
    return result;
  }

  static python_matrix_iterator *obtain(MatrixType &m)
  {
    std::auto_ptr<python_matrix_iterator> it(new python_matrix_iterator);
    it->m_iterator1 = m.begin1();
    it->m_iterator2 = it->m_iterator1.begin();
    return it.release();
  }
};




template <typename MatrixType, typename ResultGenerator>
struct python_matrix_iterator<MatrixType, ResultGenerator, mpl::true_>
{
  typedef
    typename ResultGenerator::result_type 
    result_type;

  ResultGenerator m_generator;

  typename MatrixType::iterator m_iterator;

  python_matrix_iterator *iter()
  {
    return this;
  }

  result_type next()
  {
    if (m_iterator == m_iterator().end())
    {
      PyErr_SetNone(PyExc_StopIteration);
      throw python::error_already_set();
    }
    else
      return m_generator.apply(m_iterator++);
  }

  static python_matrix_iterator *obtain(MatrixType &m)
  {
    std::auto_ptr<python_matrix_iterator> it(new python_matrix_iterator);
    it->m_iterator = m.begin();
    return it.release();
  }
};




// element accessors ----------------------------------------------------------
struct slice_info
{
  int m_start;
  int m_end;
  int m_step;
  int m_sliceLength;
};




void translateSlice(PyObject *slice_or_constant, slice_info &si, int my_length)
{
  if (PySlice_Check(slice_or_constant))
  {
    PySliceObject *slice = reinterpret_cast<PySliceObject *>(slice_or_constant);
    if (PySlice_GetIndicesEx(slice, my_length, &si.m_start, &si.m_end, 
          &si.m_step, &si.m_sliceLength) != 0)
      throw python::error_already_set();
  }
  else if (PyInt_Check(slice_or_constant))
  {
    int index = PyInt_AS_LONG(slice_or_constant);
    if (index < 0)
      index += my_length;
    if (index < 0)
      throw std::out_of_range("negative index out of bounds");
    if (index >= my_length)
      throw std::out_of_range("index out of bounds");
    si.m_start = index;
    si.m_end = index + 1;
    si.m_step = 1;
    si.m_sliceLength = 1;
  }
  else
    throw std::out_of_range("invalid index object");
}




template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct getElement
{
  static python::object apply(MatrixType &m, python::object &index)
  { 
    typedef
      typename get_corresponding_vector_type<MatrixType>::type
      vector_t;

    if (PyTuple_Check(index.ptr()))
    {
      // we have a tuple
      if (PyTuple_GET_SIZE(index.ptr()) != 2)
        throw std::out_of_range("expected tuple of size 2");

      slice_info si1, si2;
      translateSlice(PyTuple_GET_ITEM(index.ptr(), 0), si1, m.size1());
      translateSlice(PyTuple_GET_ITEM(index.ptr(), 1), si2, m.size2());

      if (si1.m_sliceLength == 1 && si2.m_sliceLength == 1)
        return python::object(m(si1.m_start, si2.m_start));
      else if (si1.m_sliceLength == 1)
        return python::object(new vector_t(row(m,si1.m_start)));
      else if (si2.m_sliceLength == 1)
        return python::object(new vector_t(column(m,si2.m_start)));
      else
      {
        return python::object(
            new MatrixType(project(m,
              ublas::slice(si1.m_start, si1.m_step, si1.m_sliceLength),
              ublas::slice(si2.m_start, si2.m_step, si2.m_sliceLength))));
      }
    }
    else
    {
      slice_info si;
      translateSlice(index.ptr(), si, m.size1());

      if (si.m_sliceLength == 1)
        return python::object(
            new vector_t(
              row(m, si.m_start)));
      else
        return python::object(
            new MatrixType(project(m,
                ublas::slice(si.m_start, si.m_step, si.m_sliceLength),
                ublas::slice(0, 1, m.size2())
                )));
    }
  }
};




template <typename MatrixType>
struct getElement<MatrixType, mpl::true_>
{
  static python::object
    apply(MatrixType &m, python::object &index)
  { 
    slice_info si;
    translateSlice(index.ptr(), si, m.size());

    if (si.m_sliceLength == 1)
      return python::object(m(si.m_start));
    else
      return python::object(
          new MatrixType(project(m, ublas::slice(si.m_start, si.m_step, si.m_sliceLength))));
  }
};




template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct setElement
{
  static void apply(MatrixType &m, python::object &index,
      python::object &new_value)
  { 
    python::extract<typename MatrixType::value_type> new_scalar(new_value);
    python::extract<MatrixType> new_matrix(new_value);

    typedef 
      typename get_corresponding_vector_type<MatrixType>::type
      vector_type;

    python::extract<vector_type> new_vector(new_value);

    if (PyTuple_Check(index.ptr()))
    {
      // we have a tuple
      if (PyTuple_GET_SIZE(index.ptr()) != 2)
        throw std::out_of_range("expected tuple of size 2");

      slice_info si1, si2;
      translateSlice(PyTuple_GET_ITEM(index.ptr(), 0), si1, m.size1());
      translateSlice(PyTuple_GET_ITEM(index.ptr(), 1), si2, m.size2());

      if (si1.m_sliceLength == 1 && si2.m_sliceLength == 1 && new_scalar.check())
      {
        m(si1.m_start, si2.m_start) = new_scalar();
      }
      else if (si1.m_sliceLength == 1 && new_vector.check())
      {
        vector_type new_vec = new_vector();

        if (new_vec.size() != m.size2())
          throw std::out_of_range("submatrix is wrong size for assignment");

        row(m,si1.m_start) = new_vec;
      }
      else if (si2.m_sliceLength == 1 && new_vector.check())
      {
        vector_type new_vec = new_vector();

        if (new_vec.size() != m.size1())
          throw std::out_of_range("submatrix is wrong size for assignment");

        column(m,si2.m_start) = new_vec;
      }
      else
      {
        MatrixType new_mat = new_matrix();
        if (int(new_mat.size1()) != si1.m_sliceLength || int(new_mat.size2()) != si2.m_sliceLength)
          throw std::out_of_range("submatrix is wrong size for assignment");

        project(m,
              ublas::slice(si1.m_start, si1.m_step, si1.m_sliceLength),
              ublas::slice(si2.m_start, si2.m_step, si2.m_sliceLength)) = new_mat;
      }
    }
    else
    {
      slice_info si;
      translateSlice(index.ptr(), si, m.size1());

      if (si.m_sliceLength == 1 && new_vector.check())
      {
        vector_type new_vec = new_vector();

        if (new_vec.size() != m.size2())
          throw std::out_of_range("submatrix is wrong size for assignment");

        row(m,si.m_start) = new_vec;
      }
      else
      {
        MatrixType new_mat = new_matrix();

        if (int(new_mat.size1()) != si.m_sliceLength || new_mat.size2() != m.size2())
          throw std::out_of_range("submatrix is wrong size for assignment");

        project(m,
            ublas::slice(si.m_start, si.m_step, si.m_sliceLength),
            ublas::slice(0, 1, m.size2())) = new_matrix();
      }
    }
  }
};




template <typename MatrixType>
struct setElement<MatrixType, mpl::true_>
{
  static void apply(MatrixType &m, python::object &index, 
    python::object &new_value)
  { 
    python::extract<typename MatrixType::value_type> new_scalar(new_value);
    python::extract<MatrixType> new_matrix(new_value);

    slice_info si;
    translateSlice(index.ptr(), si, m.size());

    if (si.m_sliceLength == 1 && new_scalar.check())
      m(si.m_start) = new_scalar();
    else
      project(m, ublas::slice(si.m_start, si.m_step, si.m_sliceLength)) = 
        new_matrix();
  }
};




// specialty constructors -----------------------------------------------------
template <typename MatrixType>
MatrixType *getIdentityMatrix(unsigned n)
{
  return new MatrixType(
      ublas::identity_matrix<typename MatrixType::value_type>(n));
}




template <typename MatrixType>
MatrixType *getFilledMatrix(
    typename MatrixType::size_type size1, 
    typename MatrixType::size_type size2, 
    const typename MatrixType::value_type &value)
{
  std::auto_ptr<MatrixType> mat(new MatrixType(size1, size2));
  
  // cannot use iterators here since sparse types are initially "empty"
  for (unsigned i = 0; i < mat->size1(); i++)
    for (unsigned j = 0; j < mat->size1(); j++)
      (*mat)(i,j) = value;

  return mat.release();
}




template <typename MatrixType>
MatrixType *getFilledVector(
    typename MatrixType::size_type size1, 
    const typename MatrixType::value_type &value)
{
  std::auto_ptr<MatrixType> mat(new MatrixType(size1));

  // cannot use iterators here since sparse types are initially "empty"
  for (unsigned i = 0; i < mat->size(); i++)
    (*mat)(i) = value;

  return mat.release();
}




// operators ------------------------------------------------------------------
template <typename MatrixType>
MatrixType negateOp(const MatrixType &m) { return -m; }

template <typename MatrixType>
MatrixType plusOp(const MatrixType &m1, const MatrixType &m2) { return m1+m2; }

template <typename MatrixType>
MatrixType minusOp(const MatrixType &m1, const MatrixType &m2) { return m1-m2; }

template <typename MatrixType>
MatrixType plusAssignOp(MatrixType &m1, const MatrixType &m2) { return m1 += m2; }

template <typename MatrixType>
MatrixType minusAssignOp(MatrixType &m1, const MatrixType &m2) { return m1 -= m2; }

template <typename MatrixType, typename Scalar>
MatrixType scalarTimesOp(const MatrixType &m1, const Scalar &s) { return m1 * s; }

template <typename MatrixType, typename Scalar>
MatrixType scalarDivideOp(const MatrixType &m1, const Scalar &s) { return m1 / s; }

template <typename MatrixType, typename Scalar>
MatrixType scalarTimesAssignOp(MatrixType &m1, const Scalar &s) { return m1 *= s; }

template <typename MatrixType, typename Scalar>
MatrixType scalarDivideAssignOp(MatrixType &m1, const Scalar &s) { return m1 /= s; }




// wrapper for stuff that is common to vectors and matrices -------------------
template <typename PythonClass, typename WrappedClass>
void exposeUfuncs(PythonClass &pyc, WrappedClass)
{
  def("conjugate", conjugateWrapper<WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
  def("real", realWrapper<WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
  def("imaginary", imagWrapper<WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());

#define MAKE_UNARY_UFUNC(f) \
  def(#f, ufuncs::UnaryUfuncApplicator<WrappedClass, \
      ufuncs::SimpleFunctionAdapter_##f<typename WrappedClass::value_type> >::apply, \
      python::return_value_policy<python::manage_new_object>());
  MAKE_UNARY_UFUNC(cos);
  MAKE_UNARY_UFUNC(cosh);
  MAKE_UNARY_UFUNC(exp);
  MAKE_UNARY_UFUNC(log);
  MAKE_UNARY_UFUNC(log10);
  MAKE_UNARY_UFUNC(sin);
  MAKE_UNARY_UFUNC(sinh);
  MAKE_UNARY_UFUNC(sqrt);
  MAKE_UNARY_UFUNC(tan);
  MAKE_UNARY_UFUNC(tanh);
#undef MAKE_UNARY_UFUNC
}




template <typename PythonClass, typename WrappedClass>
void exposeElementWiseBehavior(PythonClass &pyc, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;
  pyc
    .def("typecode", &typecode<WrappedClass>)
    .def("copy", copyNew<WrappedClass>, 
        python::return_value_policy<python::manage_new_object>())

    .def("__getitem__", getElement<WrappedClass>::apply)
    .def("__setitem__", setElement<WrappedClass>::apply)

    // stringification
    .def("__str__", &stringify<WrappedClass>)

    // unary negation
    .def("__neg__", negateOp<WrappedClass>)

    // matrix - matrix
    .def("__add__", plusOp<WrappedClass>)
    .def("__sub__", minusOp<WrappedClass>)
    .def("__iadd__", plusAssignOp<WrappedClass>)
    .def("__isub__", minusAssignOp<WrappedClass>)

    // scalar - matrix

    // this is redundant for doubles, but that's faster than yet
    // another partial specialization. Boost.Python doesn't mind
    // double definitions.
    .def("__mul__", scalarTimesOp<WrappedClass, double>)
    .def("__mul__", scalarTimesOp<WrappedClass, value_type>)

    .def("__rmul__", scalarTimesOp<WrappedClass, double>)
    .def("__rmul__", scalarTimesOp<WrappedClass, value_type>)

    .def("__div__", scalarDivideOp<WrappedClass, double>)
    .def("__div__", scalarDivideOp<WrappedClass, value_type>)

    .def("__imul__", scalarTimesAssignOp<WrappedClass, double>)
    .def("__imul__", scalarTimesAssignOp<WrappedClass, value_type>)

    .def("__idiv__", scalarDivideAssignOp<WrappedClass, double>)
    .def("__idiv__", scalarDivideAssignOp<WrappedClass, value_type>)
    ;

  exposeUfuncs(pyc, WrappedClass());
}




template <typename PythonClass, typename WrappedClass>
void exposeIterator(PythonClass &pyc, const std::string &python_typename, WrappedClass)
{
  typedef 
    python_matrix_iterator<WrappedClass, value_iterator_result_generator<WrappedClass> >
    value_iterator;

  typedef 
    python_matrix_iterator<WrappedClass, key_iterator_result_generator<WrappedClass> >
    key_iterator;

  pyc
    .def("__iter__", &value_iterator::obtain,
        python::return_value_policy<python::manage_new_object,
        python::return_internal_reference<> >())
    .def("indices", &key_iterator::obtain,
        python::return_value_policy<python::manage_new_object,
        python::return_internal_reference<> >())
    ;

  class_<key_iterator>
    ((python_typename + "KeyIterator").c_str(), python::no_init)
    .def("next", &key_iterator::next)
    .def("__iter__", &key_iterator::iter,
        python::return_self<>())
    ;

  class_<python_matrix_iterator<WrappedClass, value_iterator_result_generator<WrappedClass> > >
    ((python_typename + "ValueIterator").c_str(), python::no_init)
    .def("next", &value_iterator::next)
    .def("__iter__", &value_iterator::iter,
        python::return_self<>())
    ;
}




// vector wrapper -------------------------------------------------------------
template <typename PythonClass, typename WrappedClass>
void exposeVectorConcept(PythonClass &pyclass, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;

  exposeElementWiseBehavior(pyclass, WrappedClass());

  // inner and outer products
  def("innerproduct", inner_prodWrapper<WrappedClass, WrappedClass>::apply);
  def("outerproduct", outer_prodWrapper<WrappedClass, WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
}




template <typename WrappedClass>
void exposeVectorType(WrappedClass, const std::string &python_typename, const std::string &python_eltypename)
{
  std::string total_typename = python_typename + python_eltypename;
  class_<WrappedClass> pyclass(total_typename.c_str());

  pyclass
    .def(python::init<typename WrappedClass::size_type>())
    .add_property("shape", getShape<WrappedClass>::apply, setShape<WrappedClass>::apply)
    .def("swap", &WrappedClass::swap)
    .def("getFilledMatrix", &getFilledVector<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .staticmethod("getFilledMatrix")
    ;

  exposeVectorConcept(pyclass, WrappedClass());
  exposeIterator(pyclass, total_typename, WrappedClass());
}




// matrix wrapper -------------------------------------------------------------
template <typename PythonClass, typename WrappedClass>
void exposeMatrixConcept(PythonClass &pyclass, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;

  exposeElementWiseBehavior(pyclass, WrappedClass());

  // products
  def("matrixmultiply", prodMatMatWrapper<WrappedClass, WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
  def("matrixmultiply", prodMatVecWrapper<WrappedClass, ublas::vector<value_type> >::apply,
      python::return_value_policy<python::manage_new_object>());
  def("matrixmultiply", prodMatVecWrapper<ublas::vector<value_type>, WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());

  def("transpose", transposeMatrix<WrappedClass>,
      python::return_value_policy<python::manage_new_object>());
  def("hermite", hermiteMatrix<WrappedClass>,
      python::return_value_policy<python::manage_new_object>());
}




template <typename PythonClass, typename WrappedClass, typename ValueType>
void exposeMatrixConvertersForValueType(PythonClass &pyclass, WrappedClass, ValueType)
{
  pyclass
    .def(python::init<const ublas::matrix<ValueType> &>())
    .def(python::init<const ublas::coordinate_matrix<ValueType> &>())
    .def(python::init<const ublas::compressed_matrix<ValueType> &>())
    ;
}




template <typename PythonClass, typename WrappedClass, 
  typename _ValueType = typename WrappedClass::value_type>
struct exposeMatrixConverters { };

template <typename PythonClass, typename WrappedClass>
struct exposeMatrixConverters<PythonClass, WrappedClass, std::complex<double> >
{
  static void apply(PythonClass &pyclass, WrappedClass)
  {
    exposeMatrixConvertersForValueType(pyclass, WrappedClass(), double());
    exposeMatrixConvertersForValueType(pyclass, WrappedClass(), std::complex<double>());

    pyclass
      .def(python::init<
          const managed_symmetric_adaptor<ublas::compressed_matrix<double> > &>())
      .def(python::init<
          const managed_hermitian_adaptor<ublas::compressed_matrix<std::complex<double> > > &>())
      .def(python::init<
          const managed_symmetric_adaptor<ublas::coordinate_matrix<double> > &>())
      .def(python::init<
          const managed_hermitian_adaptor<ublas::coordinate_matrix<std::complex<double> > > &>())
      ;
  }
};




template <typename PythonClass, typename WrappedClass>
struct exposeMatrixConverters<PythonClass, WrappedClass, double>
{
  static void apply(PythonClass &pyclass, WrappedClass)
  {
    exposeMatrixConvertersForValueType(pyclass, WrappedClass(), double());

    pyclass
      .def(python::init<
          const managed_symmetric_adaptor<ublas::compressed_matrix<double> > &>())
      .def(python::init<
          const managed_symmetric_adaptor<ublas::coordinate_matrix<double> > &>())
      ;
  }
};




template <typename WrappedClass>
void exposeMatrixType(WrappedClass, const std::string &python_typename, const std::string &python_eltypename)
{
  std::string total_typename = python_typename + python_eltypename;
  class_<WrappedClass> pyclass(total_typename.c_str());

  pyclass
    .def(python::init<typename WrappedClass::size_type, 
        typename WrappedClass::size_type>())
    .add_property("shape", getShape<WrappedClass>::apply, setShape<WrappedClass>::apply)
    .def("swap", &WrappedClass::swap)

    // special constructors
    .def("getIdentityMatrix", &getIdentityMatrix<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .def("getFilledMatrix", &getFilledMatrix<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .staticmethod("getIdentityMatrix")
    .staticmethod("getFilledMatrix")
    ;

  exposeMatrixConcept(pyclass, WrappedClass());
  exposeIterator(pyclass, total_typename, WrappedClass());
  exposeMatrixConverters<class_<WrappedClass>, WrappedClass>::apply(pyclass, WrappedClass());
}




#define EXPOSE_ALL_TYPES \
  exposeAll(double(), "Float64"); \
  exposeAll(std::complex<double>(), "Complex64"); \



} // private namespace
