cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(CTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_program(CLANGXX NAMES clang++)
if(CLANGXX)
  set(CMAKE_CXX_COMPILER ${CLANGXX})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
  message(FATAL_ERROR "clang++ not found. Please install clang.")
endif()

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*main.cpp$")

add_executable(CTorch src/main.cpp ${SOURCE_FILES})
target_include_directories(CTorch PUBLIC "${PROJECT_SOURCE_DIR}/src/tensor")
add_subdirectory(tests)

add_custom_command(TARGET CTorch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/compile_commands.json
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
)
