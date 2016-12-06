###
##          Copyright Adrien Devresse 2016
## Distributed under the Boost Software License, Version 1.0.
##    (See accompanying file LICENSE_1_0.txt or copy at
##          http:##www.boost.org/LICENSE_1_0.txt)
##
## 
## Detect C++ 11 support 
## 
##  CMAKE_SUPPORT_CXX11 ( TRUE if C++11 supported, else FALSE )


include(CheckCXXSourceCompiles)


if(NOT DEFINED CMAKE_CXX_SUPPORT_CXX11)


set(_CXX_SUPPORT_TEST_SRC "\
#include <iostream>\n\
#include <vector>\n\
#include <algorithm>\n\
\
int main(){\
    std::vector<int> vec = { 1, 2 };\
    vec.emplace_back(42);\
\
    for(auto & i : vec){\
        i += 2;\
    }\
\
    std::for_each(vec.begin(), vec.end(), [](int i) {\
        std::cout << i << std::endl;\
    });\
}\
")

set(CMAKE_REQUIRED_FLAGS_OLD ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS "-std=c++11")


CHECK_CXX_SOURCE_COMPILES("${_CXX_SUPPORT_TEST_SRC}" _CXX_SUPPORT_CXX11)


set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_OLD})

set(CMAKE_CXX_SUPPORT_CXX11 ${_CXX_SUPPORT_CXX11} CACHE BOOL "support for C++11")

if(CMAKE_CXX_SUPPORT_CXX11)
    message(STATUS "Compiler ${CMAKE_CXX_COMPILER_ID} supports C++11 ")
else()
    message(STATUS "Compiler ${CMAKE_CXX_COMPILER_ID} supports C++03")
endif()

endif()
