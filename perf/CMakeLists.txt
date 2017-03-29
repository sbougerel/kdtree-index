cmake_minimum_required (VERSION 2.8.4)
project (kdtree_index_perf)

option (USE_LIBCXX "Force libc++ with Clang?" ON)

# A whole bunch of warnings we are interested in
set (SPATIAL_GNU_WARNINGS "-Wall -Wextra -Wshadow -Wcast-qual -Wconversion -Wsign-conversion -Wformat")

# Use Clang with libc++ ABI, like on Apple OS (only supports C++11)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND USE_LIBCXX)
  # Some of the flags below are to force Clang not to use anything but libc++
  # I don't think it's an issue on Darwin, but it is on Linux.
  if ("${CMAKE_SYSTEM}" MATCHES "Linux")
    message (STATUS "Building with Clang and libc++")
    set (CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -pedantic-errors")
    set (CMAKE_EXE_LINKER_FLAGS "-lc++abi")
  endif ()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPATIAL_GNU_WARNINGS}")

# Else use Clang compiler with system's default ABI
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  message(STATUS "Building with Clang")
  set (CMAKE_CXX_FLAGS "-std=c++11 -pedantic-errors")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPATIAL_GNU_WARNINGS}")

# Use Gnu compiler with libstdc++ ABI
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message (STATUS "Building with GNU")
  set (CMAKE_CXX_FLAGS "-std=c++11 -pedantic-errors")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPATIAL_GNU_WARNINGS}")

# Use Microsoft Visual Studio
elseif (MSVC)
  # Set MSVC Warning level to 4
  message (STATUS "Building with MSVC")
  if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif ()

# We are not sure which compiler you are using...
else ()
  message (WARNING "No flags have been set for your compiler")
endif ()

# The test exectuables that check correctness
add_executable (min_max min_max.cpp)

if (MSVC)
  set_target_properties (verify PROPERTIES COMPILE_FLAGS "/EHa")
endif ()
