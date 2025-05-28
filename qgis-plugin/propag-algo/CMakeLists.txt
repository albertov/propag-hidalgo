cmake_minimum_required(VERSION 3.16)

project(
  QgisPropagAlgoPlugin
  VERSION 0.1
  DESCRIPTION "A processing algorithm for QGIS to simulate wild fires"
  LANGUAGES CXX)


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Qt-specific extensions
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall")

find_package(QGIS REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets Xml Gui REQUIRED)

FIND_PATH(PROPAG_INCLUDE_DIR
    NAMES propag_host.h
    PATHS
    "${PROPAG_PREFIX_PATH}/include"
    NO_CMAKE_FIND_ROOT_PATH
  )

FIND_LIBRARY(GEOMETRY_LIBRARY
    NAMES geometry
    PATHS
    "${PROPAG_PREFIX_PATH}/lib"
)
FIND_LIBRARY(PROPAG_LIBRARY
    NAMES propag
    PATHS
    "${PROPAG_PREFIX_PATH}/lib"
)

set(LIBS
  ${QGIS_CORE_LIBRARY}
  ${PROPAG_LIBRARY}
  ${GEOMETRY_LIBRARY}
  ${QGIS_GUI_LIBRARY}
  ${QGIS_ANALYSIS_LIBRARY}
  Qt5::Core
  Qt5::Gui
  Qt5::Xml
  Qt5::Widgets
)

set(INCLUDES
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${QGIS_INCLUDE_DIR}
  ${PROPAG_INCLUDE_DIR}
)

include_directories(
  ${INCLUDES}
)

# The compiled library code is here
add_subdirectory(src)
