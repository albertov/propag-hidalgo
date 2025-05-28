#include "plugincontainer.h"

#include <dlfcn.h>
#include <iostream>
#include <stdio.h>

PluginContainer::PluginContainer(const std::string &path) : _library(0) {
  // load library
  _library = dlopen(path.c_str(), RTLD_LAZY);
  if (!_library)
    throw std::string(dlerror());
  dlerror(); // reset errors;

  // load run function
  _run = (RunFn)dlsym(_library, "FFIPropagation_run");
  const char *dlsym_error = dlerror();
  if (dlsym_error)
    throw std::string(dlerror());

  // load run function
  _rasterize_fuels =
      (RasterizeFuelsFn)dlsym(_library, "propag_rasterize_fuels");
  dlsym_error = dlerror();
  if (dlsym_error)
    throw std::string(dlerror());
};

PluginContainer::~PluginContainer() {
  if (_library) {
    dlclose(_library);
  }
};

bool PluginContainer::run(FFIPropagation propag, uintptr_t err_len, char *err) {
  return _run(propag, err_len, err);
}
bool PluginContainer::rasterize_fuels(const FFIFuelFeature *fuels,
                                      uintptr_t fuels_len,
                                      const char *fuels_crs,
                                      const GeoReference *geo_ref,
                                      uint8_t *result, char *err_msg,
                                      uintptr_t err_len) {
  return _rasterize_fuels(fuels, fuels_len, fuels_crs, geo_ref, result, err_msg,
                          err_len);
}
