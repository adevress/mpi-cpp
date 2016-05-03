# FindHPCTools
# -------------
#
# Find HPCTools
#
# Find the HPCTools Blue Brain HPC utils library
#
# Using HPCTools:
#
# ::
#
#   find_package(HPCTools REQUIRED)
#   include_directories(${HPCTools_INCLUDE_DIRS})
#   target_link_libraries(foo ${HPCTools_LIBRARIES})
#
# This module sets the following variables:
#
# ::
#
#   HPCTools_FOUND - set to true if the library is found
#   HPCTools_INCLUDE_DIRS - list of required include directories
#   HPCTools_LIBRARIES - list of libraries to be linked

#=============================================================================
# Copyright 2015 Adrien Devresse <adrien.devresse@epfl.ch>
#
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)


# UNIX paths are standard, no need to write.
find_library(HPCTools_LIBRARIES
  NAMES HPCTools
  PATHS "/usr/lib"  "/usr/lib64" "${CMAKE_PREFIX_PATH}/lib64"
  )
find_path(HPCTools_INCLUDE_DIRS
  NAMES hpctools/DistributedMatrixOperations.h
  PATHS "/usr/include"
  )


# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(HPCTools
  FOUND_VAR HPCTools_FOUND
  REQUIRED_VARS HPCTools_LIBRARIES HPCTools_INCLUDE_DIRS)
  
