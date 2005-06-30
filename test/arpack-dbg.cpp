#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <arpack.hpp>
#include <cstdlib>

namespace ublas = boost::numeric::ublas;
namespace arpack = boost::numeric::bindings::arpack;
using namespace std;

int main()
{
  const int n = 10;
  ublas::matrix<double> a(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      a(i, j) = i*j;

  arpack::results<ublas::vector<complex<double> > >  res;
  ublas::vector<double> startv = ublas::unit_vector<double>(n, 0);
  arpack::performReverseCommunication
    <ublas::matrix<double>,
    ublas::vector<complex<double> >,
    ublas::vector<double> >
    (a, 0, arpack::REGULAR_NON_GENERALIZED, 0, 3, 8, res, startv);
}
