add_library(propagplugin MODULE
  qgis_propag_algo.cpp
  qgsproviderpropag.cpp
  plugincontainer.cpp
  qgsalgorithmpropag.cpp
  qgspropagloader.cpp
)

target_link_libraries(propagplugin
  ${LIBS}
)

target_include_directories(propagplugin PUBLIC
  ${INCLUDES}
)

install(TARGETS propagplugin
        LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/qgis/plugins
)

