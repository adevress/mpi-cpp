
if(UNIT_TESTS)


include(TestHelpers)

find_package(Boost 1.41.0 REQUIRED unit_test_framework)

if(NOT Boost_USE_STATIC_LIBS)
        add_definitions(-DBOOST_TEST_DYN_LINK=TRUE)
endif()


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS} )

link_directories( ${Boost_LIBRARY_DIRS} )


## mpi Test
LIST(APPEND MpiRuntime_src "mpi_test.cpp")

include_directories(${MPI_C_INCLUDE_PATH})

add_executable(mpi_test ${MpiRuntime_src} ${MPI_CPP_HEADERS} ${MPI_CPP_HEADERS_1} ${MPI_CPP_HEADERS_2})
target_link_libraries(mpi_test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${Boost_THREAD_LIBRARIES} ${Boost_SYSTEM_LIBRARIES} ${MPI_LIBRARIES})

add_test(NAME MpiTest_unit COMMAND ${TEST_MPI_EXEC_PREFIX} ${CMAKE_CURRENT_BINARY_DIR}/mpi_test)



endif()

add_subdirectory(perf)

