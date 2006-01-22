#include <boost/python.hpp>
#include <cg.hpp>
#include <bicgstab.hpp>
#include <lu.hpp>
#include <cholesky.hpp>

#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/type.hpp>

#include <boost/numeric/ublas/triangular.hpp>

#ifdef USE_UMFPACK
#include <umfpack.hpp>
#endif // USE_UMFPACK

#ifdef USE_ARPACK
#include <arpack.hpp>

namespace arpack = boost::numeric::bindings::arpack;
#endif // USE_ARPACK

#include "meta.hpp"
#include "python_helpers.hpp"

#ifdef USE_LAPACK
#include <boost/numeric/bindings/lapack/gesdd.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/lapack/heev.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>
#endif // USE_LAPACK

/*
#include <boost/numeric/bindings/atlas/clapack.hpp>
#include <boost/numeric/ublas/lu.hpp>
*/





// shape accessors ------------------------------------------------------------
template <typename MatrixType>
inline python::object getShape(const MatrixType &m)
{ 
  return python::make_tuple(m.size1(), m.size2());
}




// wrappers -------------------------------------------------------------------
template <typename ValueType>
struct matrix_operator_wrapper : public matrix_operator<ValueType>, 
python::wrapper<matrix_operator<ValueType> >
{
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    unsigned size1() const
    {
      return this->get_override("size1")();
    }
    unsigned size2() const
    {
      return this->get_override("size2")();
    }
    void apply(const vector_type &before, vector_type &after) const
    {
      this->get_override("apply")(boost::cref(before), 
                                  boost::ref(after));
    }
};




// ublas_matrix_operator ------------------------------------------------------
template <typename MatrixType>
static ublas_matrix_operator<MatrixType> *makeMatrixOperator(const MatrixType &mat)
{
  return new ublas_matrix_operator<MatrixType>(mat);
}




struct ublas_matrix_operator_exposer
{
  template <typename MatrixType>
  void expose(const std::string &python_mattype, MatrixType) const
  {
    typedef 
      typename MatrixType::value_type
      value_type;
    typedef 
      typename matrix_operator<value_type>::vector_type
      vector_type;

    python::class_<ublas_matrix_operator<MatrixType>, 
    python::bases<matrix_operator<value_type> > >
      (("MatrixOperator" + python_mattype).c_str(),
       python::init<const MatrixType &>()[python::with_custodian_and_ward<1,2>()]);
    python::def("makeMatrixOperator", makeMatrixOperator<MatrixType>,
        python::return_value_policy<
        python::manage_new_object,
        python::with_custodian_and_ward_postcall<0, 1> >());
  }
};




// cholesky_exposer -----------------------------------------------------------
struct cholesky_exposer
{
public:
  template <typename MatrixType>
  void expose(const std::string &python_mattype, MatrixType) const
  {
    python::def("cholesky", cholesky::cholesky<MatrixType, MatrixType>,
        python::return_value_policy<python::manage_new_object>());
  }
};





// lu -------------------------------------------------------------------------
/*
 UBLAS builtin lu is dog-slow.
template <typename MatrixType>
python::object luWrapper(const MatrixType &a)
{
  using namespace ublas;

  typedef 
    typename strip_symmetric_wrappers<MatrixType>::type
    result_type;
  typedef
    permutation_matrix<unsigned>
    permut_type;

  std::auto_ptr<MatrixType> a_copy(new MatrixType(a));
  std::auto_ptr<permut_type> permut_ptr(new permut_type(a.size1()));

  axpy_lu_factorize(*a_copy, *permut_ptr);

  std::auto_ptr<result_type> l(new result_type(
        triangular_adaptor<MatrixType, unit_lower>(*a_copy)));
  std::auto_ptr<result_type> u(new result_type(
      triangular_adaptor<MatrixType, upper>(*a_copy)));

  int sign = 1;
  python::list py_permut;
  for (unsigned i = 0; i < permut_ptr->size(); i++)
  {
    // FIXME: BUG...
    py_permut.append((*permut_ptr)[i]);
    // FIXME: prove that this is right.
    if ((*permut_ptr)[i] != i) 
      sign *= -1;
  }
  
  // FIXME: pyobject_from_new_ptr
  //python::object py_result = python::make_tuple(python::object(l.get()), 
  //python::object(u.get()), py_permut, sign);
  l.release();
  u.release();

  return py_result;
}
*/




/*
My LU is still slow, but faster than the UBLAS builtin.
*/
template <typename MatrixType>
PyObject *luWrapper(const MatrixType &a)
{
  typedef MatrixType result_type;
  boost::tuple<result_type *, result_type *, std::vector<unsigned> *, int> result = 
    lu::lu<MatrixType, result_type, result_type>(a);
  std::auto_ptr<result_type> l(result.get<0>()), u(result.get<1>());
  std::auto_ptr<std::vector<unsigned> > permut_ptr(result.get<2>());
  std::vector<unsigned> &permut = *permut_ptr;

  python::list py_permut;
  for (unsigned i = 0; i < permut.size(); i++)
    py_permut.append(permut[i]);
  
  return Py_BuildValue("(NNOi)",
                       pyobject_from_new_ptr(l.release()),
                       pyobject_from_new_ptr(u.release()), 
                       py_permut.ptr(),
                       result.get<3>());
}




/*
 couldn't find atlas CLAPACK 
template <typename MatrixType>
python::object luWrapper(const MatrixType &a)
{
  using namespace boost::numeric::bindings::atlas;

  typedef 
    typename strip_symmetric_wrappers<MatrixType>::type
    result_type;
  typedef
    ublas::vector<int>
    permut_type;

  std::auto_ptr<MatrixType> a_copy(new MatrixType(a));
  std::auto_ptr<permut_type> permut_ptr(new permut_type(a.size1()));

  lu_factor(*a_copy, *permut_ptr);

  std::auto_ptr<result_type> l(new result_type(
        ublas::triangular_adaptor<MatrixType, ublas::unit_lower>(*a_copy)));
  std::auto_ptr<result_type> u(new result_type(
        ublas::triangular_adaptor<MatrixType, ublas::upper>(*a_copy)));

  int sign = 1;
  python::list py_permut;
  for (unsigned i = 0; i < permut_ptr->size(); i++)
  {
    // FIXME: BUG...
    py_permut.append((*permut_ptr)[i]);
    // FIXME: prove that this is right.
    if ((*permut_ptr)[i] != i) 
      sign *= -1;
  }
  
  python::object py_result = python::make_tuple(
  pyobject_from_new_ptr(l.get()), 
  pyobject_from_new_ptr(u.get()), py_permut, sign);
  l.release();
  u.release();

  return py_result;
}
*/




// svd ------------------------------------------------------------------------
#ifdef USE_LAPACK
template <typename ValueType>
static PyObject *svdWrapper(const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  
  fortran_mat a_copy(a), u(a.size1(), a.size1()), vt(a.size2(), a.size2());
  ublas::vector<double> s(std::min(a.size1(), a.size2()));
  
  int ierr = boost::numeric::bindings::lapack::gesdd(a_copy, s, u, vt);
  if (ierr < 0)
    throw std::runtime_error("invalid argument to gesdd");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");

  typedef ublas::matrix<ValueType> mat;
  typedef ublas::vector<ValueType> vec;
  return Py_BuildValue("(NNN)", 
                       pyobject_from_new_ptr(new mat(u)),
                       pyobject_from_new_ptr(new vec(s)), 
                       pyobject_from_new_ptr(new mat(vt)));
}
#endif // USE_LAPACK




// eigenvectors ---------------------------------------------------------------
#ifdef USE_LAPACK
void _Heigenvectors_backend(char jobz, char uplo, 
			   ublas::matrix<double, ublas::column_major> &a, 
			   ublas::vector<double> &w) 
{
  int ierr = boost::numeric::bindings::lapack::syev(jobz, uplo, a, w, 
						boost::numeric::bindings::lapack::optimal_workspace());
  if (ierr < 0)
    throw std::runtime_error("invalid argument to syev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");
}

void _Heigenvectors_backend(char jobz, char uplo, 
			    ublas::matrix<std::complex<double>, ublas::column_major> &a, 
			    ublas::vector<double> &w) 
{
  int ierr = boost::numeric::bindings::lapack::heev(jobz, uplo, a, w, 
						    boost::numeric::bindings::lapack::optimal_workspace());
  if (ierr < 0)
    throw std::runtime_error("invalid argument to heev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");
}

template <typename ValueType>
static PyObject *HeigenvectorsWrapper(bool get_vectors, bool upper, 
                                      const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  typedef ublas::vector<double> vec;
  
  fortran_mat a_copy(a);
  std::auto_ptr<vec> w(new vec(a.size1()));
  
  _Heigenvectors_backend(get_vectors ? 'V' : 'N',
			 upper ? 'U' : 'L',
			 a_copy, *w);

  typedef ublas::matrix<ValueType> mat;
  return Py_BuildValue("(NN)",
                       pyobject_from_new_ptr(new mat(a_copy)), 
                       pyobject_from_new_ptr(w.release()));
}




template <typename ValueType>
static PyObject *eigenvectorsWrapper(unsigned get_left_vectors, 
                                     unsigned get_right_vectors,
                                     const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  typedef ublas::vector<typename helpers::complexify<ValueType>::type > eval_vec;
  typedef ublas::matrix<typename helpers::complexify<ValueType>::type, ublas::column_major> evec_mat;
  
  int const n = a.size1();
  fortran_mat a_copy(a);
  std::auto_ptr<eval_vec> w(new eval_vec(a.size1()));

  std::auto_ptr<evec_mat> vl, vr;
  if (get_left_vectors)
    vl = std::auto_ptr<evec_mat>(new evec_mat(n, n));
  if (get_right_vectors)
    vr = std::auto_ptr<evec_mat>(new evec_mat(n, n));

  int ierr = boost::numeric::bindings::lapack::geev(a_copy, *w, vl.get(), vr.get(),
						    boost::numeric::bindings::lapack::optimal_workspace());

  if (ierr < 0)
    throw std::runtime_error("invalid argument to geev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");

  typedef ublas::matrix<typename helpers::complexify<ValueType>::type> cmat;
  typedef ublas::vector<ValueType> vec;

  if (get_left_vectors)
  {
    if (get_right_vectors)
      return Py_BuildValue("(NNN)",
                           pyobject_from_new_ptr(w.release()),
                           pyobject_from_new_ptr(new cmat(*vl)),
                           pyobject_from_new_ptr(new cmat(*vr)));
    else
      return Py_BuildValue("(NNs)",
                           pyobject_from_new_ptr(w.release()),
                           pyobject_from_new_ptr(new cmat(*vl)),
                           NULL);
  }
  else
  {
    if (get_right_vectors)
      return Py_BuildValue("(NsN)",
                           pyobject_from_new_ptr(w.release()),
                           NULL,
                           pyobject_from_new_ptr(new cmat(*vr)));
    else
      return Py_BuildValue("(Nss)",
                           pyobject_from_new_ptr(w.release()),
                           NULL,
                           NULL);
  }
}
#endif // USE_LAPACK




// ----------------------------------------------------------------------------
template <typename ValueType>
void exposeSpecialAlgorithms(ValueType)
{
#ifdef USE_LAPACK
  python::def("lu", luWrapper<ublas::matrix<ValueType> >);
  python::def("singular_value_decomposition", svdWrapper<ValueType>);
  python::def("Heigenvectors", HeigenvectorsWrapper<ValueType>);
  python::def("eigenvectors", eigenvectorsWrapper<ValueType>);
#endif // USE_LAPACK
}





// matrix operators -----------------------------------------------------------
template <typename ValueType>
static void exposeMatrixOperators(const std::string &python_eltname, ValueType)
{
  {
    typedef matrix_operator<ValueType> wrapped_type;
    python::class_<matrix_operator_wrapper<ValueType>, noncopyable >
      (("MatrixOperator"+python_eltname).c_str())
      .add_property("shape", &getShape<wrapped_type>)
      .def("size1", python::pure_virtual(&wrapped_type::size1))
      .def("size2", python::pure_virtual(&wrapped_type::size2))
      .def("apply", &wrapped_type::apply)
      ;
  }

  {
    typedef algorithm_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
      python::bases<matrix_operator<ValueType> >,
      noncopyable>
      (("AlgorithmMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("debug_level", &wrapped_type::getDebugLevel, &wrapped_type::setDebugLevel)
      .add_property("last_iteration_count", &wrapped_type::getLastIterationCount)
      ;
  }

  {
    typedef iterative_solver_matrix_operator<ValueType> wrapped_type;

    python::class_<wrapped_type, 
      python::bases<algorithm_matrix_operator<ValueType> >,
      noncopyable >
      (("IterativeSolverMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("max_iterations", &wrapped_type::getMaxIterations, &wrapped_type::setMaxIterations)
      .add_property("tolerance", &wrapped_type::getTolerance, &wrapped_type::setTolerance)
      ;
  }

  {
    python::class_<identity_matrix_operator<ValueType>, 
    python::bases<matrix_operator<ValueType> > >
      (("IdentityMatrixOperator"+python_eltname).c_str(), 
       python::init<unsigned>());
  }

  {
    python::class_<composite_matrix_operator<ValueType>, 
    python::bases<matrix_operator<ValueType> > >
      (("CompositeMatrixOperator"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<ValueType> &, 
         const matrix_operator<ValueType> &>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    python::class_<sum_of_matrix_operators<ValueType>, 
    python::bases<matrix_operator<ValueType> > >
      (("SumOfMatrixOperators"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<ValueType> &, 
         const matrix_operator<ValueType> &>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    python::class_<scalar_multiplication_matrix_operator<ValueType>, 
    python::bases<matrix_operator<ValueType> > >
      (("ScalarMultiplicationMatrixOperator"+python_eltname).c_str(), 
       python::init<ValueType, unsigned>());
  }

  {
    typedef cg::cg_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<iterative_solver_matrix_operator<ValueType> > >
      (("CGMatrixOperator"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<ValueType> &, 
         const matrix_operator<ValueType>&, 
         unsigned, double>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    typedef bicgstab::bicgstab_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
      python::bases<iterative_solver_matrix_operator<ValueType> > >
      (("BiCGSTABMatrixOperator"+python_eltname).c_str(), 
       python::init<
       const matrix_operator<ValueType> &, 
       const matrix_operator<ValueType>&, 
       unsigned, double>()
       [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

#ifdef USE_UMFPACK
  { 
    typedef umfpack::umfpack_matrix_operator<ValueType> wrapped_type;
    typedef 
      python::class_<wrapped_type, 
      python::bases<algorithm_matrix_operator<ValueType> >, boost::noncopyable>    
        wrapper_type;

    wrapper_type pyclass(("UMFPACKMatrixOperator"+python_eltname).c_str(), 
       python::init<const typename wrapped_type::matrix_type &>()
       [python::with_custodian_and_ward<1,2>()]);
  }
#endif // USE_UMFPACK
}




// arpack ---------------------------------------------------------------------
#ifdef USE_ARPACK
template <typename ResultsType>
static typename ResultsType::value_container::iterator beginRitzValues(ResultsType &res)
{
  return res.m_ritz_values.begin();
}

template <typename ResultsType>
static typename ResultsType::value_container::iterator endRitzValues(ResultsType &res)
{
  return res.m_ritz_values.end();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator beginRitzVectors(ResultsType &res)
{
  return res.m_ritz_vectors.begin();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator endRitzVectors(ResultsType &res)
{
  return res.m_ritz_vectors.end();
}

template <typename ValueType, typename RealType>
arpack::results<ublas::vector<std::complex<RealType> > > *wrapArpack(
      const matrix_operator<ValueType> &op, 
      const matrix_operator<ValueType> *m,
      arpack::arpack_mode mode,
      std::complex<RealType> spectral_shift,
      int number_of_eigenvalues,
      int number_of_arnoldi_vectors,
      arpack::which_eigenvalues which_e,
      RealType tolerance,
      int max_iterations
      )
{
  typedef arpack::results<ublas::vector<std::complex<RealType> > > results_type;
  std::auto_ptr<results_type> results(new results_type());
  ublas::vector<ValueType> start_vector = ublas::unit_vector<ValueType>(op.size1(), 0);
  try
  {
    arpack::performReverseCommunication(
      op, m, mode, spectral_shift, 
      number_of_eigenvalues, number_of_arnoldi_vectors,
      *results, start_vector,
      which_e, tolerance, max_iterations);
  }
  catch (std::exception &ex)
  {
    std::cerr << ex.what() << std::endl;
    throw;
  }
  return results.release();
}




template <typename ValueType>
static void exposeArpack(const std::string &python_valuetypename, ValueType)
{
  typedef typename arpack::results<ublas::vector<ValueType> > results_type;
  typedef typename ublas::type_traits<ValueType>::real_type real_type;

  python::class_<results_type>
    (("ArpackResults"+python_valuetypename).c_str())
    .add_property("RitzValues", 
        python::range(beginRitzValues<results_type>, endRitzValues<results_type>))
    .add_property("RitzVectors", 
        python::range(beginRitzVectors<results_type>, endRitzVectors<results_type>))
    ;

  python::def("runArpack", wrapArpack<ValueType, real_type>,
              python::return_value_policy<python::manage_new_object>());
}
#endif // USE_ARPACK



// library support queries ----------------------------------------------------
bool has_blas() { return USE_BLAS; }
bool has_lapack() { return USE_LAPACK; }
bool has_arpack() { return USE_ARPACK; }
bool has_umfpack() { return USE_UMFPACK; }




// main -----------------------------------------------------------------------
BOOST_PYTHON_MODULE(_operation)
{
  exposeMatrixOperators("Float64", double());
  exposeMatrixOperators("Complex64", std::complex<double>());

  // expose bicgstab only for real-valued matrices
  
  // expose complex adaptor only for real-valued matrices
  {
    typedef double ValueType;
    typedef complex_matrix_operator_adaptor<ValueType> wrapped_type;
    python::class_<wrapped_type, 
      python::bases<matrix_operator<std::complex<ValueType> > > >
      ("ComplexMatrixOperatorAdaptorFloat64", 
       python::init<
       const matrix_operator<ValueType> &, 
       const matrix_operator<ValueType> &>()
       [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  exposeForAllMatrices(ublas_matrix_operator_exposer());

#ifdef USE_ARPACK
  python::enum_<arpack::which_eigenvalues>("arpack_which_eigenvalues")
    .value("SMALLEST_MAGNITUDE", arpack::SMALLEST_MAGNITUDE)
    .value("LARGEST_MAGNITUDE", arpack::LARGEST_MAGNITUDE)
    .value("SMALLEST_REAL_PART", arpack::SMALLEST_REAL_PART)
    .value("LARGEST_REAL_PART", arpack::LARGEST_REAL_PART)
    .value("SMALLEST_IMAGINARY_PART", arpack::SMALLEST_IMAGINARY_PART)
    .value("LARGEST_IMAGINARY_PART", arpack::LARGEST_IMAGINARY_PART)
    .export_values();

  python::enum_<arpack::arpack_mode>("arpack_mode")
    .value("REGULAR_NON_GENERALIZED", arpack::REGULAR_NON_GENERALIZED)
    .value("REGULAR_GENERALIZED", arpack::REGULAR_GENERALIZED)
    .value("SHIFT_AND_INVERT_GENERALIZED", arpack::SHIFT_AND_INVERT_GENERALIZED)
    .export_values();

  exposeArpack("Float64", double());
  exposeArpack("Complex64", std::complex<double>());
#endif // USE_ARPACK

  exposeSpecialAlgorithms(double());
  exposeSpecialAlgorithms(std::complex<double>());

  exposeForAllMatrices(cholesky_exposer());

  python::def("has_blas", has_blas);
  python::def("has_lapack", has_lapack);
  python::def("has_arpack", has_arpack);
  python::def("has_umfpack", has_umfpack);
}




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
