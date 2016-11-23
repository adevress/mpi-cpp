#ifndef MPI_CPP_CONFIG_HPP
#define MPI_CPP_CONFIG_HPP


// auto detect C++1 support
#if __cplusplus >=  201103L  
#ifndef _MPI_CPP_USE_CXX11
#define _MPI_CPP_USE_CXX11 1
#else
#define _MPI_CPP_USE_CXX11 0
#endif
#endif

// Enforce restricted compatibility with MPI 1.1
#ifndef MPI_FORCE_LEGACY
#define MPI_FORCE_LEGACY 0
#endif 


#if _MPI_CPP_USE_CXX11 == 0
#	include <boost/shared_ptr.hpp>
#else
#	include <memory>
#endif


#if MPI_FORCE_LEGACY 
#	include "mpi_1_1_compatibility.hpp"
#endif



namespace mpi {




namespace util {

// utilise boost for shared pointer in case of C++ < 2011 
#if _MPI_CPP_USE_CXX11 == 0
using namespace boost;
#else
template<typename Type>
using shared_ptr = std::shared_ptr<Type>;
#endif


} // util


} // mpi

#endif // MPI_CPP_CONFIG_HPP

