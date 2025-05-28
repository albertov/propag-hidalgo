add_library(propagplugin MODULE
  qgis_propag_algo.cpp
  qgsproviderpropag.cpp
  plugin_base.cpp
  plugincontainer.cpp
)
add_library(propagalgoplugin MODULE
  qgsalgorithmpropag.cpp
  plugin_base.cpp
  qgspropagloader.cpp
)

target_link_libraries(propagplugin
  libdl.so
  ${LIBS}
)
target_link_libraries(propagalgoplugin
  ${LIBS}
)

target_include_directories(propagplugin PUBLIC
  ${INCLUDES}
)
target_include_directories(propagalgoplugin PUBLIC
  ${INCLUDES}
)

install(TARGETS propagalgoplugin
        LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/lib
)
install(TARGETS propagplugin
        LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/qgis/plugins
)

