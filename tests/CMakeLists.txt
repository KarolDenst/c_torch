# Specify the path to the googletest source directory
set(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/external/googletest)

# Add GoogleTest directory with binary directory explicitly specified
add_subdirectory(${GOOGLETEST_DIR} ${CMAKE_BINARY_DIR}/googletest)

# Include GoogleTest directories for LSP/IDE to find headers
include_directories(${GOOGLETEST_DIR}/googletest/include)
include_directories(${GOOGLETEST_DIR}/googlemock/include)

# Add test executable
add_executable(tests test_main.cpp)

# Link test executable against GoogleTest libraries
target_link_libraries(tests gtest gtest_main)
