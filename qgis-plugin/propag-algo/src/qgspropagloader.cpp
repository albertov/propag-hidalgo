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
  return load_fuel(geo_ref, output->fuel_code);
}

bool QgsPropagLoader::load_fuel(const GeoReference *geo_ref, uint8_t *output) {
  std::vector<QByteArray> wkbs;
  std::vector<FFIFuelFeature> features;
  QgsRectangle extent(GeoReference_x0(geo_ref), GeoReference_y0(geo_ref),
                      GeoReference_x1(geo_ref), GeoReference_y1(geo_ref));

  QgsCoordinateTransformContext transformCtx =
      QgsProject::instance()->transformContext();
  QgsCoordinateReferenceSystem crs;
  if (!crs.createFromProj((const char *)geo_ref->proj)) {
    return false;
  }
  QgsCoordinateTransform fuelTransform(crs, _fuelCodes->sourceCrs(),
                                       transformCtx);
  QgsRectangle fuelExtent = fuelTransform.transformBoundingBox(extent);

  QgsFeatureRequest request(fuelExtent);
  QgsFeatureIterator it = _fuelCodes->getFeatures(request);
  QgsFeature feature;
  while (it.nextFeature(feature)) {
    QgsGeometry geom = feature.geometry();
    wkbs.push_back(geom.asWkb());
    QVariant fuelCode = feature.attribute(_fieldIdx);
    FFIFuelFeature fuel_feature(fuelCode.toInt(), NULL, 0);
    features.push_back(fuel_feature);
  };
  for (int i = 0; i < features.size(); i++) {
    features[i].geom_wkb = (const uint8_t *)wkbs[i].data();
    features[i].geom_wkb_len = wkbs[i].size();
  }
  QgsCoordinateReferenceSystem fuelCrs = _fuelCodes->sourceCrs();
  QString proj = fuelCrs.toProj();
  QByteArray proj_ba = proj.toUtf8();

  char err_c[1024];
  memset(&err_c, 0, 1024);
  bool result =
      propag_rasterize_fuels(features.data(), features.size(), proj_ba.data(),
                             geo_ref, output, err_c, 1024);
  return result;
}
