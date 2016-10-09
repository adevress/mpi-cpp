#ifndef MPI_CPP_CONFIG_HPP
#define MPI_CPP_CONFIG_HPP

#if __cplusplus >= 20011L
#ifndef _MPI_CPP_USE_CXX11
#define _MPI_CPP_USE_CXX11 1
#else
#define _MPI_CPP_USE_CXX11 0
#endif
#endif


#if _MPI_CPP_USE_CXX11 == 0

#include <boost/shared_ptr.hpp>

#else

#include <memory>

#endif

namespace mpi {


namespace util {

#if _MPI_CPP_USE_CXX11 == 0

using namespace boost;

#else

using namespace std;

#endif


} // util


} // mpi

#endif // MPI_CPP_CONFIG_HPP

