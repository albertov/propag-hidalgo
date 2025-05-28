#ifndef QGSPROPAGLOADER_H
#define QGSPROPAGLOADER_H

#include "propag_host.h"
#include "qgis.h"
#include "qgsprocessingalgorithm.h"

class QgsPropagLoader {
  QgsFeatureSource *_fuelCodes;
  int _fieldIdx;

public:
  QgsPropagLoader(QgsFeatureSource *fuelCodes, int fieldIdx)
      : _fuelCodes(fuelCodes), _fieldIdx(fieldIdx) {};

  bool load_terrain(const GeoReference *geo_ref, FFITerrain *output);

private:
  inline bool load_fuel(const GeoReference *geo_ref, uint8_t *output);
};

static bool load_terrain(void *self, const GeoReference *geo_ref,
                         FFITerrain *output) {
  return static_cast<QgsPropagLoader *>(self)->load_terrain(geo_ref, output);
};

#endif
