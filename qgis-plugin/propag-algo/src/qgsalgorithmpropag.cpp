#include "qgsalgorithmpropag.h"
#include "plugincontainer.h"
#include "qgis.h"
#include "qgspropagloader.h"

///@cond PRIVATE

/*
Qgis::ProcessingAlgorithmFlags QgsPropagAlgorithm::flags() const
{
  return QgsProcessingAlgorithm::flags() |
Qgis::ProcessingAlgorithmFlag::HideFromToolbox;
}
*/

QString QgsPropagAlgorithm::name() const { return QStringLiteral("propag"); }

QString QgsPropagAlgorithm::displayName() const {
  return QObject::tr("Simulate");
}

QStringList QgsPropagAlgorithm::tags() const {
  return QObject::tr("simulation,wildfire,forest").split(',');
}

QString QgsPropagAlgorithm::shortHelpString() const {
  return QObject::tr("TBD");
}

QgsPropagAlgorithm *QgsPropagAlgorithm::createInstance() const {
  return new QgsPropagAlgorithm();
}

void QgsPropagAlgorithm::initAlgorithm(const QVariantMap &) {
  addParameter(new QgsProcessingParameterFeatureSource(
      QStringLiteral("IGNITED_ELEMENTS"),
      QObject::tr("Initial ignited elements")));
  addParameter(new QgsProcessingParameterString(
      QStringLiteral("IGNITED_ELEMENT_TIME_FIELD"),
      QObject::tr("The name of the field of the ignited element feature that "
                  "holds the access time"),
      QVariant("time"), false, false));
  addParameter(new QgsProcessingParameterFeatureSource(
      QStringLiteral("FUEL"), QObject::tr("Fuel codes polygon layer")));
  addParameter(new QgsProcessingParameterString(
      QStringLiteral("FUEL_CODE_FIELD"),
      QObject::tr("The name of the field of the fuel code feature that "
                  "holds the fuel code"),
      QVariant("BEHAVE"), false, false));
  addParameter(new QgsProcessingParameterNumber(
      QStringLiteral("MAX_SIMULATION_MINUTES"),
      QObject::tr("Maximum simulation minutes"),
      Qgis::ProcessingNumberParameterType::Double, QVariant(5.0 * 60.0), false,
      1e-9));
  addParameter(new QgsProcessingParameterExtent(
      QStringLiteral("EXTENT"), QObject::tr("Max simulation extent"),
      QVariant(), false));
  addParameter(new QgsProcessingParameterNumber(
      QStringLiteral("CELL_SIZE_X"), QObject::tr("Cell size X"),
      Qgis::ProcessingNumberParameterType::Double, QVariant(5.0), false, 1e-9));
  addParameter(new QgsProcessingParameterNumber(
      QStringLiteral("CELL_SIZE_Y"), QObject::tr("Cell size Y"),
      Qgis::ProcessingNumberParameterType::Double, QVariant(5.0), false, 1e-9));
  addParameter(new QgsProcessingParameterRasterDestination(
      QStringLiteral("TIMES"), QObject::tr("Fire Access Times")));
  addParameter(new QgsProcessingParameterVectorDestination(
      QStringLiteral("REFERENCES"), QObject::tr("Fire References"),
      Qgis::ProcessingSourceType::VectorLine, QVariant(), true, false));
  addParameter(new QgsProcessingParameterVectorDestination(
      QStringLiteral("BLOCK_BOUNDARIES"), QObject::tr("Block boundaries"),
      Qgis::ProcessingSourceType::VectorPolygon, QVariant(), true, false));
  addParameter(new QgsProcessingParameterVectorDestination(
      QStringLiteral("GRID_BOUNDARIES"), QObject::tr("Grid boundaries"),
      Qgis::ProcessingSourceType::VectorPolygon, QVariant(), true, false));
}

QVariantMap
QgsPropagAlgorithm::processAlgorithm(const QVariantMap &parameters,
                                     QgsProcessingContext &context,
                                     QgsProcessingFeedback *feedback) {
  bool find_ref_change =
      parameters.value(QStringLiteral("REFERENCES")).isValid();

  bool generate_grid_boundaries =
      parameters.value(QStringLiteral("GRID_BOUNDARIES")).isValid();

  bool generate_block_boundaries =
      parameters.value(QStringLiteral("BLOCK_BOUNDARIES")).isValid();

  QgsFeatureSource *ignitedElements =
      parameterAsSource(parameters, "IGNITED_ELEMENTS", context);
  if (!ignitedElements) {
    throw QgsProcessingException("Invalid source layer");
  }
  QString ignited_element_time_field;
  if (parameters.value(QStringLiteral("IGNITED_ELEMENT_TIME_FIELD"))
          .isValid()) {
    ignited_element_time_field = parameterAsString(
        parameters, QStringLiteral("IGNITED_ELEMENT_TIME_FIELD"), context);
  } else {
    throw QgsProcessingException(
        QObject::tr("Invalid IGNITED_ELEMENT_TIME_FIELD"));
  }

  QgsFeatureSource *fuelCodes = parameterAsSource(parameters, "FUEL", context);
  if (!fuelCodes) {
    throw QgsProcessingException("Invalid fuel codes layer");
  }
  QString fuel_code_field;
  if (parameters.value(QStringLiteral("FUEL_CODE_FIELD")).isValid()) {
    fuel_code_field = parameterAsString(
        parameters, QStringLiteral("FUEL_CODE_FIELD"), context);
  } else {
    throw QgsProcessingException(QObject::tr("Invalid FUEL_CODE_FIELD"));
  }

  bool hasXValue = parameters.value(QStringLiteral("CELL_SIZE_X")).isValid();
  bool hasYValue = parameters.value(QStringLiteral("CELL_SIZE_Y")).isValid();
  double xSize, ySize;
  if ((hasXValue && !hasYValue) || (!hasXValue && hasYValue)) {
    throw QgsProcessingException(QObject::tr(
        "Either set both X and Y cell size values or keep both as 'Not set'."));
  } else if (hasXValue && hasYValue) {
    xSize =
        parameterAsDouble(parameters, QStringLiteral("CELL_SIZE_X"), context);
    ySize =
        parameterAsDouble(parameters, QStringLiteral("CELL_SIZE_Y"), context);
  } else {
    throw QgsProcessingException(QObject::tr("Invalid CELL_SIZE"));
  }
  if (xSize <= 0) {
    throw QgsProcessingException(QObject::tr("Invalid CELL_SIZE_X"));
  }
  if (ySize <= 0) {
    throw QgsProcessingException(QObject::tr("Invalid CELL_SIZE_Y"));
  }

  QgsRectangle extent;
  if (parameters.value(QStringLiteral("EXTENT")).isValid()) {
    extent = parameterAsExtent(parameters, QStringLiteral("EXTENT"), context);
  } else {
    throw QgsProcessingException(QObject::tr("Invalid EXTENT"));
  }
  // Use the extent CRS as the target CRS
  QgsCoordinateReferenceSystem crs =
      parameterAsExtentCrs(parameters, QStringLiteral("EXTENT"), context);

  double max_time;
  if (parameters.value(QStringLiteral("MAX_SIMULATION_MINUTES")).isValid()) {
    max_time =
        parameterAsDouble(parameters, QStringLiteral("MAX_SIMULATION_MINUTES"),
                          context) *
        60.0;
  } else {
    throw QgsProcessingException(QObject::tr("Invalid MAX_SIMULATION_MINUTES"));
  }
  if (max_time <= 0) {
    throw QgsProcessingException(
        QObject::tr("MAX_SIMULATION_MINUTES must be > 0"));
  }

  GeoReference geo_ref;
  QString proj = crs.toProj();
  QByteArray proj_ba = proj.toUtf8();
  if (!GeoReference_south_up(extent.xMinimum(), extent.yMinimum(),
                             extent.xMaximum(), extent.yMaximum(), xSize, ySize,
                             proj_ba.data(), &geo_ref)) {
    throw QgsProcessingException(
        QObject::tr("Could not create a valid GeoReference with given "
                    "CELL_SIZE_? and EXTENT"));
  }

  Settings settings(geo_ref, max_time);

  int fuelIdx = fuelCodes->fields().indexOf(fuel_code_field);
  if (fuelIdx == -1) {
    throw QgsProcessingException(
        QObject::tr("FUEL source does not have a FUEL_CODE_FIELD"));
  }

  PluginContainer plugin("libpropag.so");

  QgsPropagLoader loader(&plugin, fuelCodes, fuelIdx);
  FFITerrainLoader terrain_loader(&load_terrain, &loader);

  std::vector<QByteArray> wkbs;
  std::vector<FFITimeFeature> features;

  QgsCoordinateTransformContext transformCtx =
      QgsProject::instance()->transformContext();
  QgsCoordinateTransform ignitedElementsTransform(
      crs, ignitedElements->sourceCrs(), transformCtx);
  QgsRectangle ignitedElementsExtent =
      ignitedElementsTransform.transformBoundingBox(extent);
  QgsFeatureRequest request(ignitedElementsExtent);
  QgsFeatureIterator it = ignitedElements->getFeatures();
  QgsFeature feature;
  int timeIdx = ignitedElements->fields().indexOf(ignited_element_time_field);
  if (timeIdx == -1) {
    throw QgsProcessingException(QObject::tr(
        "IGNITED_ELEMENTS source does not have a IGNITED_ELEMENT_TIME_FIELD"));
  }
  while (it.nextFeature(feature)) {
    QgsGeometry geom = feature.geometry();
    wkbs.push_back(geom.asWkb());
    QVariant timeValue = feature.attribute(timeIdx);
    FFITimeFeature time_feature(timeValue.toDouble() * 60.0, NULL, 0);
    features.push_back(time_feature);
  };
  for (int i = 0; i < features.size(); i++) {
    features[i].geom_wkb = (const uint8_t *)wkbs[i].data();
    features[i].geom_wkb_len = wkbs[i].size();
  }
  QString ie_proj = ignitedElements->sourceCrs().toProj();
  QByteArray ie_proj_ba = ie_proj.toUtf8();

  const QString outputFile =
      parameterAsOutputLayer(parameters, QStringLiteral("TIMES"), context);
  QByteArray outputFile_ba = outputFile.toUtf8();
  const char *output_path = outputFile_ba.data();

  QString refsOutputFile;
  if (find_ref_change) {
    refsOutputFile = parameterAsOutputLayer(
        parameters, QStringLiteral("REFERENCES"), context);
  };
  QByteArray refsOutputFile_ba = refsOutputFile.toUtf8();
  const char *refs_output_path = refsOutputFile_ba.data();

  QString blockBoundariesOutputFile;
  if (generate_block_boundaries) {
    blockBoundariesOutputFile = parameterAsOutputLayer(
        parameters, QStringLiteral("BLOCK_BOUNDARIES"), context);
  };
  QByteArray blockBoundariesOutputFile_ba = blockBoundariesOutputFile.toUtf8();
  const char *block_boundaries_output_path =
      blockBoundariesOutputFile_ba.data();

  QString gridBoundariesOutputFile;
  if (generate_grid_boundaries) {
    gridBoundariesOutputFile = parameterAsOutputLayer(
        parameters, QStringLiteral("GRID_BOUNDARIES"), context);
  };
  QByteArray gridBoundariesOutputFile_ba = gridBoundariesOutputFile.toUtf8();
  const char *grid_boundaries_output_path = gridBoundariesOutputFile_ba.data();

  FFIPropagation propagation(
      settings, output_path, refs_output_path, block_boundaries_output_path,
      grid_boundaries_output_path, features.data(), features.size(),
      ie_proj_ba.data(), terrain_loader);

  char err_c[1024];
  memset(&err_c, 0, 1024);
  if (!plugin.run(propagation, 1024, err_c)) {
    throw QgsProcessingException(err_c);
  }

  QVariantMap outputs;
  outputs.insert(QStringLiteral("TIMES"), outputFile);
  if (find_ref_change)
    outputs.insert(QStringLiteral("REFERENCES"),
                   refsOutputFile + "|layername=fire_references");
  if (generate_block_boundaries) {
    outputs.insert(QStringLiteral("BLOCK_BOUNDARIES"),
                   blockBoundariesOutputFile + "|layername=blocks");
  }
  if (generate_grid_boundaries) {
    outputs.insert(QStringLiteral("GRID_BOUNDARIES"),
                   blockBoundariesOutputFile + "|layername=grids");
  }
  return outputs;
}

///@endcond
