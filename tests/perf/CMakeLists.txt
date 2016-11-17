
if(PERF_TESTS)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

find_package(Boost 1.41.0 REQUIRED system thread unit_test_framework)


if(NOT Boost_USE_STATIC_LIBS)
        add_definitions(-DBOOST_TEST_DYN_LINK=TRUE)
endif()


## pingpong_perf_tests Test
LIST(APPEND mpicpp_pingpong_perf_src "pingpong_perf.cpp")

add_definitions( -std=c++11 )

add_executable(mpicpp_pingpong_perf_bin ${mpicpp_pingpong_perf_src})
target_link_libraries(mpicpp_pingpong_perf_bin ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${Boost_SYSTEM_LIBRARIES} ${Boost_THREAD_LIBRARIES} ${MPI_LIBRARIES})

endif()
