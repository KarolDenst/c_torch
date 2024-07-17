set(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/external/googletest)
add_subdirectory(${GOOGLETEST_DIR} ${CMAKE_BINARY_DIR}/googletest)
include_directories(${GOOGLETEST_DIR}/googletest/include)
include_directories(${GOOGLETEST_DIR}/googlemock/include)

include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE TEST_SOURCES ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(tests ${TEST_SOURCES} ${SOURCE_FILES})

target_link_libraries(tests gtest gtest_main)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
