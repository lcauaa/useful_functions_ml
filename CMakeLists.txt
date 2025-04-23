# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Define the project name and version
project(usefulfuncs VERSION 1.0)

# Set the C standard
set(CMAKE_C_STANDARD 99)

# Enable AVX2 support
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")

# Enable OpenMP if available
find_package(OpenMP REQUIRED)

# Specify where to find the header files
include_directories(include)

# Define the source files for the library
set(SRC_FILES
    src/usefulfuncs.c
    src/matrixops.c
)

# Define where to put the compiled library
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Optionally, define where to put the compiled binaries
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Build the static library
add_library(usefulfuncs STATIC ${SRC_FILES})

# Build the shared library
add_library(usefulfuncs_shared SHARED ${SRC_FILES})

# Specify the library version for shared library (optional)
set_target_properties(usefulfuncs_shared PROPERTIES VERSION ${PROJECT_VERSION})

# Link OpenMP to your library and executable
target_link_libraries(usefulfuncs OpenMP::OpenMP_C)

# If you want to create an executable for testing
add_executable(test_matrixops tests/test_matrixops.c)

# Link the test executable with the OpenMP-enabled library
target_link_libraries(test_matrixops usefulfuncs OpenMP::OpenMP_C)

# Optionally, enable testing and add a test (with CTest)
enable_testing()
add_test(NAME SimpleTest COMMAND test_matrixops)

# Installation rules (optional)
install(TARGETS usefulfuncs usefulfuncs_shared DESTINATION lib)
install(FILES include/usefulfuncs.h include/matrixops.h DESTINATION include)
