project(mpi-cpp)
cmake_minimum_required(VERSION 2.8)

enable_testing()

set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")

option(UNIT_TESTS "Enable or disable unit tests execution" TRUE)
option(MPI_CPP_DISABLE_INSTALL "Disable the installation for this component" FALSE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
      ${PROJECT_SOURCE_DIR}/CMake/portability
      ${PROJECT_SOURCE_DIR}/CMake/packages
      ${PROJECT_SOURCE_DIR}/CMake/config)

set(Boost_NO_BOOST_CMAKE TRUE)

include(BlueGenePortability)
include(DefineInstallationPaths)
include(ReleaseDebugAutoFlags)

find_package(MPI)


file(GLOB MPI_CPP_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/mpi-cpp/*)
file(GLOB MPI_CPP_HEADERS_1 ${CMAKE_CURRENT_SOURCE_DIR}/include/mpi-cpp/*/*)
file(GLOB MPI_CPP_HEADERS_2 ${CMAKE_CURRENT_SOURCE_DIR}/include/mpi-cpp/*/*/*)


include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${BOOST_INCLUDE_DIR} )

if(NOT MPI_CPP_DISABLE_INSTALL)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/mpi-cpp DESTINATION ${INCLUDE_INSTALL_DIR}/)
endif()

add_subdirectory(tests)

