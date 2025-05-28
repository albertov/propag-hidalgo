#ifndef PLUGINCONTAINER_H
#define PLUGINCONTAINER_H

#include "plugin_base.h"

class PluginContainer {
public:
  PluginContainer(const std::string &path);
  ~PluginContainer();
  bool run(FFIPropagation, uintptr_t, char *);
  bool rasterize_fuels(const FFIFuelFeature *fuels, uintptr_t fuels_len,
                       const char *fuels_crs, const GeoReference *geo_ref,
                       uint8_t *result, char *err_msg, uintptr_t err_len);

private:
  void *_library;
  propagation_run_t _run;
  rasterize_fuels_t _rasterize_fuels;
};

#endif // PLUGINCONTAINER_H
