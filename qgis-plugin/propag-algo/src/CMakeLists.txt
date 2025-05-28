add_library(propagalgoplugin MODULE
  qgis_propag_algo.cpp
  qgsalgorithmpropag.cpp
  qgsproviderpropag.cpp
)

target_link_libraries(propagalgoplugin
  ${LIBS}
)

target_include_directories(propagalgoplugin PUBLIC
  ${INCLUDES}
)

install(TARGETS propagalgoplugin
        LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/qgis/plugins
)

