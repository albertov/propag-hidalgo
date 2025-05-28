#ifndef QGSPROPAGLOADER_H
#define QGSPROPAGLOADER_H

#include "propag_host.h"

class QgsPropagLoader {
public:
  bool load_terrain(const GeoReference *geo_ref, FFITerrain *output);
};

#endif
