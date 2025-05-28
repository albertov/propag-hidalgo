#include "qgsalgorithmpropag.h"
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
  return QObject::tr("Wildfire Propagator");
}

QStringList QgsPropagAlgorithm::tags() const {
  return QObject::tr("simulation").split(',');
}

QString QgsPropagAlgorithm::group() const { return QObject::tr("Simulators"); }

QString QgsPropagAlgorithm::groupId() const {
  return QStringLiteral("simulators");
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
      QObject::tr("Initial ignited elements layer")));
  addParameter(new QgsProcessingParameterString(
      QStringLiteral("IGNITED_ELEMENT_TIME_FIELD"),
      QObject::tr("The name of the field of the ignited element feature that "
                  "holds the access time"),
      QVariant("time"), false, false));
  addParameter(new QgsProcessingParameterBoolean(
      QStringLiteral("GENERATE_REFS"),
      QObject::tr("Generate references vector layer"), false));
  addParameter(new QgsProcessingParameterNumber(
      QStringLiteral("MAX_SIMULATION_MINUTES"),
      QObject::tr("Maximum simulation minutes"),
      Qgis::ProcessingNumberParameterType::Double, QVariant(5.0 * 60.0), false,
      1e-9));
  addParameter(new QgsProcessingParameterCrs(
      QStringLiteral("CRS"),
      QObject::tr("Output Coordinate Reference System")));
  addParameter(new QgsProcessingParameterNumber(
      QStringLiteral("CELL_SIZE_X"), QObject::tr("Cell size X"),
      Qgis::ProcessingNumberParameterType::Double, QVariant(5.0), false, 1e-9));
  addParameter(new QgsProcessingParameterNumber(
      QStringLiteral("CELL_SIZE_Y"), QObject::tr("Cell size Y"),
      Qgis::ProcessingNumberParameterType::Double, QVariant(5.0), false, 1e-9));
  addParameter(new QgsProcessingParameterExtent(
      QStringLiteral("EXTENT"), QObject::tr("Extent"), QVariant(), false));
  addParameter(new QgsProcessingParameterRasterDestination(
      QStringLiteral("TIMES"), QObject::tr("Access time output raster")));
}

QVariantMap
QgsPropagAlgorithm::processAlgorithm(const QVariantMap &parameters,
                                     QgsProcessingContext &context,
                                     QgsProcessingFeedback *feedback) {
  const QString outputFile =
      parameterAsOutputLayer(parameters, QStringLiteral("TIMES"), context);

  QgsFeatureSource *ignitedElements =
      parameterAsSource(parameters, "IGNITED_ELEMENTS", context);
  if (!ignitedElements) {
    throw QgsProcessingException("Invalid source layer");
  }

  const bool find_ref_change =
      parameterAsBoolean(parameters, QStringLiteral("GENERATE_REFS"), context);

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

  QString ignited_element_time_field;
  if (parameters.value(QStringLiteral("IGNITED_ELEMENT_TIME_FIELD"))
          .isValid()) {
    ignited_element_time_field = parameterAsString(
        parameters, QStringLiteral("IGNITED_ELEMENT_TIME_FIELD"), context);
  } else {
    throw QgsProcessingException(
        QObject::tr("Invalid IGNITED_ELEMENT_TIME_FIELD"));
  }

  QgsCoordinateReferenceSystem crs;
  if (parameters.value(QStringLiteral("CRS")).isValid()) {
    crs = parameterAsCrs(parameters, QStringLiteral("CRS"), context);
  } else {
    throw QgsProcessingException(QObject::tr("Invalid CRS"));
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

  Settings settings(geo_ref, max_time, find_ref_change);

  QByteArray outputFile_ba = outputFile.toUtf8();
  const char *output_path = outputFile_ba.data();

  QgsPropagLoader loader;
  FFITerrainLoader terrain_loader(&load_terrain, &loader);

  std::vector<QByteArray> wkbs;
  std::vector<FFITimeFeature> features;
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
    QByteArray const *wkb = &wkbs[wkbs.size() - 1];
    QVariant timeValue = feature.attribute(timeIdx);
    FFITimeFeature time_feature(timeValue.toDouble() * 60.0,
                                (const uint8_t *)wkb->data(), wkb->size());
    features.push_back(time_feature);
  };
  QString ie_proj = ignitedElements->sourceCrs().toProj();
  QByteArray ie_proj_ba = ie_proj.toUtf8();

  FFIPropagation propagation(settings, output_path, features.data(),
                             features.size(), ie_proj_ba.data(),
                             terrain_loader);

  std::string err(1024, 0);
  const char *err_c = err.c_str();
  if (!FFIPropagation_run(propagation, &err_c)) {
    throw QgsProcessingException(err_c);
  }

  QVariantMap outputs;
  outputs.insert(QStringLiteral("TIMES"), outputFile);
  return outputs;
}

///@endcond
