cmake_minimum_required(VERSION 3.20)

project(deep_learning)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # because the default gcc compiler in my computer doesn't support format,
    # so i have to use libc++ instead of gcc's stdlibc++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -fexperimental-library") # Latter for the std::execution
endif()

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# Create interface library since it's header-only
add_library(metann INTERFACE)

# Get all .hpp files from specified directories
file(GLOB_RECURSE META_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/metann/*.hpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.hpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.hpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/policy/*.hpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.hpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/eval/*.hpp"
)

# Create the include directory structure
set(METANN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/metann")
file(MAKE_DIRECTORY ${METANN_INCLUDE_DIR})

# Copy headers to include/metann directory maintaining directory structure
foreach(HEADER ${META_HEADERS})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${HEADER})
    get_filename_component(DIR_PATH ${REL_PATH} DIRECTORY)
    file(COPY ${HEADER} DESTINATION "${METANN_INCLUDE_DIR}/${DIR_PATH}")
endforeach()

# Set include directories for the library
target_include_directories(metann
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add header files to the library for IDE support
target_sources(metann
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include"
    FILES ${INSTALLED_HEADERS}
)

# Set up installation rules
include(GNUInstallDirs)

# Install header files maintaining directory structure
install(DIRECTORY "${METANN_INCLUDE_DIR}"
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the target
install(TARGETS metann
    EXPORT metannTargets
    FILE_SET HEADERS
)

# Export the targets
install(EXPORT metannTargets
    FILE metannTargets.cmake
    NAMESPACE metann::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metann
)

include(CTest)
add_subdirectory(Tests)
add_subdirectory(examples)

# include_directories(utils)
# include_directories(policy)
# include_directories(data)

# add_subdirectory(chapter4)
# add_subdirectory(chapter5)
# add_subdirectory(chapter6)
# add_subdirectory(chapter7)
# add_subdirectory(chapter8)
