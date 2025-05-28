#ifndef INTERFACE_H
#define INTERFACE_H

#include "propag_host.h"
#include <string>

typedef bool (*propagation_run_t)(FFIPropagation, uintptr_t, char *);
typedef bool (*rasterize_fuels_t)(const FFIFuelFeature *, uintptr_t,
                                  const char *, const GeoReference *, uint8_t *,
                                  char *, uintptr_t);

#endif // INTERFACE_H
