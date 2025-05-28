#ifndef QGSPROPAGLOADER_H
#define QGSPROPAGLOADER_H

#include "plugincontainer.h"
#include "propag_host.h"
#include "qgis.h"
#include "qgsprocessingalgorithm.h"
#include <qgsexception.h>

class QgsPropagLoader {
  QgsFeatureSource *_fuelCodes;
  int _fieldIdx;
  PluginContainer *_plugin;

public:
  QgsPropagLoader(PluginContainer *plugin, QgsFeatureSource *fuelCodes,
                  int fieldIdx)
      : _plugin(plugin), _fuelCodes(fuelCodes), _fieldIdx(fieldIdx) {};

  bool load_terrain(const GeoReference *geo_ref, FFITerrain *output);

private:
  inline bool load_fuel(const GeoReference *geo_ref, uint8_t *output);
};

static bool load_terrain(void *self, const GeoReference *geo_ref,
                         FFITerrain *output) {
  try {
    return static_cast<QgsPropagLoader *>(self)->load_terrain(geo_ref, output);
  } catch (const std::exception &ex) {
    std::cerr << "load_terrain crashed: " << ex.what() << std::endl;
    return false;
  } catch (const QgsException &ex) {
    std::cerr << "load_terrain crashed: " << ex.what().toUtf8().toStdString()
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "load_terrain crashed" << std::endl;
    return false;
  }
};

#endif
