//
// Copyright (c) 2004-2007
// Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#include <numeric>
#include <functional>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <cg.hpp>
#include <bicgstab.hpp>
#include <lu.hpp>
#include <cholesky.hpp>

#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/type.hpp>

// tools ----------------------------------------------------------------------
#include "meta.hpp"
#include "python_helpers.hpp"




#ifdef USE_FFTW
#include <fftw3.h>




namespace
{
  using namespace boost::python;
  namespace ublas = boost::numeric::ublas;




  template <class T>
  class fftw_ptr
  {
    private:
      T *m_data;

    public:
      explicit fftw_ptr(T *data)
        : m_data(data)
      { }

      fftw_ptr(fftw_ptr &src)
        : m_data(src.m_data)
      { src.m_data = 0; }

      fftw_ptr &operator=(fftw_ptr &src)
      { 
        m_data = src.m_data;
        src.m_data = 0; 
      }

      ~fftw_ptr()
      { fftw_free(m_data); }

      T *get()
      { return m_data; }
  };




  // plan ---------------------------------------------------------------------
  template<class In, class Out>
  class plan : boost::noncopyable
  {
    private:
      fftw_plan m_plan;
      fftw_ptr<In> m_in;
      fftw_ptr<Out> m_out;
      unsigned m_len_in, m_len_out;

    public:
      plan(fftw_plan plan, fftw_ptr<In> in, fftw_ptr<Out> out, 
          unsigned len_in, unsigned len_out)
        : m_plan(plan), m_in(in), m_out(out), 
        m_len_in(len_in), m_len_out(len_out)
      { }

      ~plan()
      { fftw_destroy_plan(m_plan); }

      void execute() const
      { fftw_execute(m_plan); }
      void print() const
      { fftw_print_plan(m_plan); }

      void get_in(ublas::vector<In> &in)
      {
        in.resize(m_len_in);
        std::copy(m_in.get(), m_in.get()+m_len_in, in.begin());
      }

      void set_in(const ublas::vector<In> &in)
      {
        if (in.size() != m_len_in)
          PYTHON_ERROR(ValueError, "in vector has the wrong size");

        std::copy(in.begin(), in.end(), m_in.get());
      }

      void get_out(ublas::vector<Out> &out)
      {
        out.resize(m_len_out);
        std::copy(m_out.get(), m_out.get()+m_len_out, out.begin());
      }
  };




  template <class In, class Out>
  void expose_plan()
  {
    typedef plan<In, Out> cl;
    class_<cl, boost::noncopyable>("Plan", no_init)
      .def("execute", &cl::execute)
      .def("print", &cl::print)
      .def("get_in", &cl::get_in)
      .def("set_in", &cl::set_in)
      .def("get_out", &cl::get_out)
      ;
  }




  // dft ----------------------------------------------------------------------
  typedef std::complex<double> complex;

  plan<complex, complex> *plan_dft(object py_lengths, int sign, unsigned flags)
  {
    std::vector<int> lengths;
    copy(
        stl_input_iterator<int>(py_lengths), 
        stl_input_iterator<int>(),
        back_inserter(lengths));

    int total_length = std::accumulate(lengths.begin(), lengths.end(), 
        1, std::multiplies<double>());
    fftw_ptr<complex> in(reinterpret_cast<complex *>(
        fftw_malloc(total_length*sizeof(complex))));
    fftw_ptr<complex> out(reinterpret_cast<complex *>(
          fftw_malloc(total_length*sizeof(complex))));

    fftw_plan result = fftw_plan_dft(
        lengths.size(), lengths.data(),
        reinterpret_cast<fftw_complex *>(in.get()), 
        reinterpret_cast<fftw_complex *>(out.get()),
        sign, flags);

    if (result == NULL)
      PYTHON_ERROR(RuntimeError, "fftw dft planning failed")

    return new plan<complex, complex>(
        result,
        in, out,
        total_length, total_length);
  }




  plan<double, complex> *plan_dft_r2c(object py_lengths, unsigned flags)
  {
    std::vector<int> lengths;
    copy(
        stl_input_iterator<int>(py_lengths), 
        stl_input_iterator<int>(),
        back_inserter(lengths));
    if (lengths.size() == 0)
      PYTHON_ERROR(RuntimeError, "a rank 0 r2c transform is invalid");

    int total_length = std::accumulate(lengths.begin(), lengths.end(), 
        1, std::multiplies<double>());
    int real_length = std::accumulate(lengths.begin(), lengths.end()-1, 
        1, std::multiplies<double>());
    real_length *= (lengths.back() / 2)+1;

    fftw_ptr<double> in(reinterpret_cast<double *>(
        fftw_malloc(real_length*sizeof(double))));
    fftw_ptr<complex> out(reinterpret_cast<complex *>(
          fftw_malloc(total_length*sizeof(complex))));

    fftw_plan result = fftw_plan_dft_r2c(
        lengths.size(), lengths.data(),
        in.get(), 
        reinterpret_cast<fftw_complex *>(out.get()),
        flags);

    if (result == NULL)
      PYTHON_ERROR(RuntimeError, "fftw dft planning failed");

    return new plan<double, complex>(
        result,
        in, out,
        real_length, total_length);
  }




  plan<complex, double> *plan_dft_c2r(object py_lengths, unsigned flags)
  {
    std::vector<int> lengths;
    copy(
        stl_input_iterator<int>(py_lengths), 
        stl_input_iterator<int>(),
        back_inserter(lengths));
    if (lengths.size() == 0)
      PYTHON_ERROR(RuntimeError, "a rank 0 r2c transform is invalid");

    int total_length = std::accumulate(lengths.begin(), lengths.end(), 
        1, std::multiplies<double>());
    int real_length = std::accumulate(lengths.begin(), lengths.end()-1, 
        1, std::multiplies<double>());
    real_length *= (lengths.back() / 2)+1;

    fftw_ptr<complex> in(reinterpret_cast<complex *>(
        fftw_malloc(total_length*sizeof(complex))));
    fftw_ptr<double> out(reinterpret_cast<double *>(
          fftw_malloc(real_length*sizeof(double))));

    fftw_plan result = fftw_plan_dft_c2r(
        lengths.size(), lengths.data(),
        reinterpret_cast<fftw_complex *>(in.get()),
        out.get(), 
        flags);

    if (result == NULL)
      PYTHON_ERROR(RuntimeError, "fftw dft planning failed");

    return new plan<complex, double>(
        result,
        in, out,
        total_length, real_length);
  }




  plan<double, double> *plan_r2r(object py_lengths, object py_kinds, unsigned flags)
  {
    std::vector<int> lengths;
    copy(
        stl_input_iterator<int>(py_lengths), 
        stl_input_iterator<int>(),
        back_inserter(lengths));

    std::vector<fftw_r2r_kind> kinds;
    copy(
        stl_input_iterator<fftw_r2r_kind>(py_kinds), 
        stl_input_iterator<fftw_r2r_kind>(),
        back_inserter(kinds));

    if (lengths.size() != kinds.size())
      PYTHON_ERROR(RuntimeError, "kinds do not have the same size as lengths");

    int total_length = std::accumulate(lengths.begin(), lengths.end(), 
        1, std::multiplies<double>());
    fftw_ptr<double> in(reinterpret_cast<double *>(
        fftw_malloc(total_length*sizeof(double))));
    fftw_ptr<double> out(reinterpret_cast<double *>(
          fftw_malloc(total_length*sizeof(double))));

    fftw_plan result = fftw_plan_r2r(
        lengths.size(), lengths.data(),
        in.get(), out.get(), 
        kinds.data(), flags);

    if (result == NULL)
      PYTHON_ERROR(RuntimeError, "fftw r2r planning failed");

    return new plan<double, double>(
        result,
        in, out,
        total_length, total_length);
  }




  // wisdom -------------------------------------------------------------------
  void import_system_wisdom()
  {
    if (fftw_import_system_wisdom() != 1)
      PYTHON_ERROR(RuntimeError, "system wisdom could not be read")
  }

  void import_wisdom_from_string(const std::string &str)
  {
    fftw_import_wisdom_from_string(str.c_str());
  }

  std::string export_wisdom_to_string()
  {
    fftw_ptr<char> wisdom(fftw_export_wisdom_to_string());
    return std::string(wisdom.get());
  }
}
#endif




BOOST_PYTHON_MODULE(_fft)
{
#ifdef USE_FFTW
  using boost::python::arg;

#define EXPOSE_FFTW_CONSTANT(NAME)\
  scope().attr(#NAME) = FFTW_##NAME;

  EXPOSE_FFTW_CONSTANT(FORWARD);
  EXPOSE_FFTW_CONSTANT(BACKWARD);

  EXPOSE_FFTW_CONSTANT(NO_TIMELIMIT);

  EXPOSE_FFTW_CONSTANT(MEASURE);
  //EXPOSE_FFTW_CONSTANT(DESTROY_INPUT);
  EXPOSE_FFTW_CONSTANT(UNALIGNED);
  EXPOSE_FFTW_CONSTANT(CONSERVE_MEMORY);
  EXPOSE_FFTW_CONSTANT(EXHAUSTIVE);
  //EXPOSE_FFTW_CONSTANT(PRESERVE_INPUT);
  EXPOSE_FFTW_CONSTANT(PATIENT);
  EXPOSE_FFTW_CONSTANT(ESTIMATE);

  expose_plan<complex, complex>();
  expose_plan<complex, double>();
  expose_plan<double, double>();

  def("plan_dft", plan_dft,
      (arg("dimensions"), arg("sign"), arg("flags")=0),
      return_value_policy<manage_new_object>()
     );
  def("plan_dft_r2c", plan_dft_r2c,
      (arg("dimensions"), arg("flags")=0),
      return_value_policy<manage_new_object>()
     );
  def("plan_dft_c2r", plan_dft_c2r,
      (arg("dimensions"), arg("flags")=0),
      return_value_policy<manage_new_object>()
     );
  def("plan_dft_r2r", plan_r2r,
      (arg("dimensions"), arg("kinds"), arg("flags")=0),
      return_value_policy<manage_new_object>()
     );

  def("import_system_wisdom", import_system_wisdom);
  def("import_wisdom", import_wisdom_from_string,
      (arg("wisdom_str")));
  def("export_wisdom", import_wisdom_from_string);
  def("forget_wisdom", fftw_forget_wisdom);
#endif
}
