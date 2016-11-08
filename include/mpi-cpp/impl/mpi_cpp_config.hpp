#ifndef MPI_CPP_CONFIG_HPP
#define MPI_CPP_CONFIG_HPP

#if __cplusplus >=  201103L  
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

template<typename Type>
using shared_ptr = std::shared_ptr<Type>;

#endif


} // util


} // mpi

#endif // MPI_CPP_CONFIG_HPP

