#ifndef QGSPROPAGLOADER_H
#define QGSPROPAGLOADER_H

#include "propag_host.h"
#include "qgis.h"
#include "qgsprocessingalgorithm.h"

class QgsPropagLoader {
  QgsFeatureSource *_fuelCodes;
  QString _fieldName;

public:
  QgsPropagLoader(QgsFeatureSource *fuelCodes, QString fieldName)
      : _fuelCodes(fuelCodes), _fieldName(fieldName) {};

  bool load_terrain(const GeoReference *geo_ref, FFITerrain *output);
};

static bool load_terrain(void *self, const GeoReference *geo_ref,
                         FFITerrain *output) {
  return static_cast<QgsPropagLoader *>(self)->load_terrain(geo_ref, output);
};

#endif
