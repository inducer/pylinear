#include <boost/python.hpp>
#include <cg.h>



using namespace boost::numeric;
using namespace boost::python;




BOOST_PYTHON_MODULE(algorithms)
{
  def("cg", cg::cg<
      ublas::matrix<std::complex<double> >, 
      ublas::vector<std::complex<double> >,
      ublas::vector<std::complex<double> > > );
}
