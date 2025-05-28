#include "qgspropagloader.h"

bool QgsPropagLoader::load_terrain(const GeoReference *geo_ref,
                                   FFITerrain *output) {
  const int len = geo_ref->width * geo_ref->height;
  for (int i = 0; i < len; i++) {
    output->fuel_code[i] = 1;
    output->d1hr[i] = 0.1;
    output->d10hr[i] = 0.1;
    output->d100hr[i] = 0.1;
    output->herb[i] = 0.1;
    output->wood[i] = 0.1;
    output->wind_speed[i] = 5.0;
    output->wind_azimuth[i] = 0.0;
    output->aspect[i] = 0.0;
    output->slope[i] = 0.0;
  }
  return true;
}
