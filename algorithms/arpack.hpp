#ifndef HEADER_SEEN_ARPACK_HPP
#define HEADER_SEEN_ARPACK_HPP




#include <matrix_operator.hpp>
#include <arpack_proto.h>
#include <helpers.hpp>
#include <vector>




namespace arpack
{
  namespace ublas = boost::numeric::ublas;

  enum which_eigenvalues { LARGEST_MAGNITUDE,
    SMALLEST_MAGNITUDE,
    LARGEST_REAL_PART,
    SMALLEST_REAL_PART,
    LARGEST_IMAGINARY_PART,
    SMALLEST_IMAGINARY_PART
  };

  namespace detail
  {
    const char *mapWhichToString(which_eigenvalues we)
    {
      switch (we)
      {
        case LARGEST_MAGNITUDE: 
          return "LM";
        case SMALLEST_MAGNITUDE: 
          return "SM";
        case LARGEST_REAL_PART: 
          return "LR";
        case SMALLEST_REAL_PART: 
          return "SR";
        case LARGEST_IMAGINARY_PART: 
          return "LI";
        case SMALLEST_IMAGINARY_PART: 
          return "SI";
        default: 
          throw std::runtime_error("arpack: invalid eigenvalue selector");
      }
    }
  }

  enum arpack_mode { 
    REGULAR_NON_GENERALIZED = 1,
    REGULAR_GENERALIZED = 2,
    SHIFT_AND_INVERT_GENERALIZED = 3
  };

  template <typename BaseType>
  struct results
  {
    typedef 
      std::vector<std::complex<BaseType> > 
      value_container;
    typedef
      std::vector<ublas::vector<std::complex<BaseType> > >
      vector_container;

    value_container m_ritz_values;
    vector_container m_ritz_vectors;
  };




  namespace detail
  {
    template <typename BaseType>
    results<BaseType> *makeResults(unsigned nconv, unsigned n, 
        BaseType *z, std::complex<BaseType> *d)
    {
      // result generation for real types
      // slightly more complicated: take care of complex conjugate pairs
      std::auto_ptr<results<BaseType> > my_results(new results<BaseType>);

      unsigned i = 0;

      while (i < nconv)
      {
        if (imag(d[i]) != 0)
        {
          // complex-conjugate pair
          if (i + 1 >= nconv)
            throw std::runtime_error("arpack: complex pair split up");

          my_results->m_ritz_values.push_back(d[i]);
          my_results->m_ritz_values.push_back(d[i+1]);

          ublas::vector<std::complex<BaseType> > ritz_vector(n);
          for (unsigned j = 0; j < n; j++)
            ritz_vector[j] = std::complex<BaseType>(z[i*n + j], z[(i+1)*n +j]);

          my_results->m_ritz_vectors.push_back(ritz_vector);
          my_results->m_ritz_vectors.push_back(conj(ritz_vector));

          i += 2;
        }
        else
        {
          // real eigenvalue, single eigenvector
          my_results->m_ritz_values.push_back(d[i]);
          ublas::vector<std::complex<BaseType> > ritz_vector(n);
          for (unsigned j = 0; j < n; j++)
            ritz_vector[j] = z[i*n + j];
          my_results->m_ritz_vectors.push_back(ritz_vector);
          i++;
        }
      }

      return my_results.release();
    }

    template <typename BaseType>
    results<BaseType> *makeResults(unsigned nconv, unsigned n, 
        std::complex<BaseType> *z, std::complex<BaseType> *d)
    {
      // result generation for complex types
      std::auto_ptr<results<BaseType> > my_results(new results<BaseType>);

      // simple: just copy everything over.
      for (unsigned i = 0; i < nconv; i++)
      {
        my_results->m_ritz_values.push_back(d[i]);

        ublas::vector<std::complex<BaseType> > ritz_vector(n);
        for (unsigned j = 0; j < n; j++)
          ritz_vector[j] = z[i*n + j];
        my_results->m_ritz_vectors.push_back(ritz_vector);
      }

      return my_results.release();
    }
  }




  template <typename ValueType>
  results<typename helpers::decomplexify<ValueType>::type> *doReverseCommunication(
      const matrix_operator<ValueType> &op, 
      const matrix_operator <ValueType> &m,
      arpack_mode mode,
      std::complex<typename helpers::decomplexify<ValueType>::type> spectral_shift,
      int number_of_eigenvalues,
      int number_of_arnoldi_vectors,
      which_eigenvalues which_e = LARGEST_MAGNITUDE,
      typename helpers::decomplexify<ValueType>::type tolerance = 1e-8,
      bool m_is_identity = false,
      int max_iterations = 0
      )
  {
    typedef 
      typename helpers::decomplexify<ValueType>::type
      base_type;
    typedef
      std::complex<base_type>
      complex_type;

    int ido = 0;
    char bmat = m_is_identity ? 'I' : 'G';
    int n = op.size1();

    if ((unsigned) n != op.size2() || (unsigned) n != m.size1() || (unsigned) n != m.size2())
      throw std::runtime_error("arpack: matrix sizes don't match.");

    char *which = const_cast<char*>(detail::mapWhichToString(which_e));

    ValueType residual[n];

    ValueType v[number_of_arnoldi_vectors * n];
    int ldv = n;

    int iparam[11];
    iparam[1-1] = 1; // exact shifts
    iparam[2-1] = 0; // unused
    iparam[3-1] = max_iterations != 0 ? max_iterations : 10000 * n;
    iparam[4-1] = 1; // block size
    iparam[5-1] = 0; // NCONV
    iparam[6-1] = 0; // IUPD, unused
    iparam[7-1] = mode;
    iparam[8-1] = 0; // NP, something to do with user-provided shifts
    iparam[9-1] = 0; // NUMOP
    iparam[10-1] = 0; // NUMOPB
    iparam[11-1] = 0; // NUMREO

    int info = 0; // we're not specifying a previous residual

    int ipntr[14];

    ValueType workd[3*n];
    int lworkl;
    if (helpers::isComplex(ValueType()))
      lworkl = 3 * number_of_arnoldi_vectors * number_of_arnoldi_vectors 
        + 5 * number_of_arnoldi_vectors;
    else
      lworkl = 3 * number_of_arnoldi_vectors * number_of_arnoldi_vectors 
        + 6 * number_of_arnoldi_vectors;

    ValueType workl[lworkl];
    double rwork[number_of_arnoldi_vectors];

    do
    {
      naupd(
          &ido,
          &bmat,
          &n,
          which,
          &number_of_eigenvalues,
          &tolerance,
          residual,
          &number_of_arnoldi_vectors,
          v,
          &ldv,
          iparam,
          ipntr,
          workd,
          workl,
          &lworkl,
          rwork,
          &info
          );

      switch (info)
      {
        case 0:
          break;
        case 1:
          throw std::runtime_error("arpack, naupd: performed max. number of iterations (1)");
        case 3:
          throw std::runtime_error("arpack, naupd: could not apply shifts (3)");
        case -1:
          throw std::runtime_error("arpack, naupd: n not positive (-1)");
        case -2:
          throw std::runtime_error("arpack, naupd: nev not positive (-2)");
        case -3:
          throw std::runtime_error("arpack, naupd: ncv <= nev or ncv > n (-3)");
        case -4:
          throw std::runtime_error("arpack, naupd: max_iterations must be bigger than zero (-4)");
        case -5:
          throw std::runtime_error("arpack, naupd: invalid WHICH (-5)");
        case -6:
          throw std::runtime_error("arpack, naupd: invalid BMAT (-6)");
        case -7:
          throw std::runtime_error("arpack, naupd: work array too short (-7)");
        case -8:
          throw std::runtime_error("arpack, naupd: LAPACK error (-8)");
        case -9:
          throw std::runtime_error("arpack, naupd: starting vector is zero (-9)");
        case -10:
          throw std::runtime_error("arpack, naupd: invalid MODE (-10)");
        case -11:
          throw std::runtime_error("arpack, naupd: MODE and BMAT don't agree (-11)");
        case -12:
          throw std::runtime_error("arpack, naupd: ISHIFT invalid (-12)");
        default:
          throw std::runtime_error("arpack, naupd: invalid naupd error code");
      }

      if (ido == -1 || ido == 1 || ido == 2)
      {
        // FIXME copying is not good
        ublas::vector<ValueType> operand, result;
        operand.resize(n);
        result.resize(n);

        ValueType *x = workd + ipntr[1-1] - 1;
        ValueType *y = workd + ipntr[2-1] - 1;

        for (int i = 0; i < n; i++)
          operand[i] = x[i];

        if (ido == 2)
          m.apply(operand, result);
        else
          op.apply(operand, result);

        for (int i = 0; i < n; i++)
          y[i] = result[i];
      }
      else if (ido == 99) /*nothing*/;
      else
        throw std::runtime_error("arpack: reverse communication failure");
    }
    while (ido != 99);

    if (max_iterations != 0)
    {
      if (max_iterations <= iparam[2])
        throw std::runtime_error("arpack: hit iteration count limit");
      max_iterations -= iparam[2];
    }

    {
      // prepare for call to neupd
      int rvec = 1;
      char howmny = 'A';
      int select[number_of_arnoldi_vectors]; // no-op
      complex_type d[number_of_eigenvalues+1];

      unsigned z_size;
      unsigned workev_size;
      if (helpers::isComplex(ValueType()))
      {
        z_size = number_of_eigenvalues;
        workev_size = 2*number_of_arnoldi_vectors;
      }
      else
      {
        z_size = number_of_eigenvalues+1;
        workev_size = 3*number_of_arnoldi_vectors;
      }

      ValueType z[z_size*n];
      int ldz = n;

      ValueType workev[workev_size];

      neupd(
          &rvec, &howmny, select, d, z, &ldz, &spectral_shift, workev,
          // naupd parameters follow
          &bmat,
          &n,
          which,
          &number_of_eigenvalues,
          &tolerance,
          residual,
          &number_of_arnoldi_vectors,
          v,
          &ldv,
          iparam,
          ipntr,
          workd,
          workl,
          &lworkl,
          rwork,
          &info
          );
      switch (info)
      {
        case 0:
          break;
        case 1:
          throw std::runtime_error("arpack, neupd: schur form could not be reordered (1)");
        case -1:
          throw std::runtime_error("arpack, neupd: n not positive (-1)");
        case -2:
          throw std::runtime_error("arpack, neupd: nev not positive (-2)");
        case -3:
          throw std::runtime_error("arpack, neupd: ncv <= nev or ncv > n (-3)");
        case -5:
          throw std::runtime_error("arpack, neupd: invalid WHICH (-5)");
        case -6:
          throw std::runtime_error("arpack, neupd: invalid BMAT (-6)");
        case -7:
          throw std::runtime_error("arpack, neupd: work array too short (-7)");
        case -8:
          throw std::runtime_error("arpack, neupd: LAPACK error (-8)");
        case -9:
          throw std::runtime_error("arpack, neupd: LAPACK _trevc failed (-9)");
        case -10:
          throw std::runtime_error("arpack, neupd: invalid MODE (-10)");
        case -11:
          throw std::runtime_error("arpack, neupd: MODE and BMAT don't agree (-11)");
        case -12:
          throw std::runtime_error("arpack, neupd: HOWNY = S invalid (-12)");
        case -13:
          throw std::runtime_error("arpack, neupd: HOWNY and RVEC don't agree (-13)");
        case -14:
          throw std::runtime_error("arpack, neupd: no eigenvalues found (-14)");
        default:
          throw std::runtime_error("arpack, neupd: invalid neupd error code");
      }

      unsigned nconv = iparam[5-1];
      return detail::makeResults(nconv, n, z, d);
    }
  }
}




#endif
