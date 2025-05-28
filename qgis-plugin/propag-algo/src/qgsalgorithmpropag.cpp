#include "qgsalgorithmpropag.h"
#include "qgspropagloader.h"
#include "qgis.h"

///@cond PRIVATE

/*
Qgis::ProcessingAlgorithmFlags QgsPropagAlgorithm::flags() const
{
  return QgsProcessingAlgorithm::flags() | Qgis::ProcessingAlgorithmFlag::HideFromToolbox;
}
*/

QString QgsPropagAlgorithm::name() const
{
  return QStringLiteral( "propag" );
}

QString QgsPropagAlgorithm::displayName() const
{
  return QObject::tr( "Wildfire Propagator" );
}

QStringList QgsPropagAlgorithm::tags() const
{
  return QObject::tr( "simulation" ).split( ',' );
}

QString QgsPropagAlgorithm::group() const
{
  return QObject::tr( "Simulators" );
}

QString QgsPropagAlgorithm::groupId() const
{
  return QStringLiteral( "simulators" );
}

QString QgsPropagAlgorithm::shortHelpString() const
{
  return QObject::tr( "TBD" );
}

QgsPropagAlgorithm *QgsPropagAlgorithm::createInstance() const
{
  return new QgsPropagAlgorithm();
}

void QgsPropagAlgorithm::initAlgorithm( const QVariantMap & )
{
  addParameter( new QgsProcessingParameterVectorLayer( QStringLiteral( "IGNITED_ELEMENTS" ), QObject::tr( "Initial ignited elements layer" ) ) );
  addParameter( new QgsProcessingParameterBoolean( QStringLiteral( "GENERATE_REFS" ), QObject::tr( "Generate references vector layer" ), false ) );
  addParameter( new QgsProcessingParameterNumber( QStringLiteral( "MAX_SIMULATION_MINUTES" ), QObject::tr( "Maximum simulation minutes" ), Qgis::ProcessingNumberParameterType::Double, QVariant(), true, 1e-9 ) );
  addParameter( new QgsProcessingParameterNumber( QStringLiteral( "CELL_SIZE_X" ), QObject::tr( "Override reference cell size X" ), Qgis::ProcessingNumberParameterType::Double, QVariant(), true, 1e-9 ) );
  addParameter( new QgsProcessingParameterNumber( QStringLiteral( "CELL_SIZE_Y" ), QObject::tr( "Override reference cell size Y" ), Qgis::ProcessingNumberParameterType::Double, QVariant(), true, 1e-9 ) );
  addParameter( new QgsProcessingParameterExtent( QStringLiteral( "EXTENT" ), QObject::tr( "Clip to extent" ), QVariant(), true ) );
  addParameter( new QgsProcessingParameterRasterDestination( QStringLiteral( "TIMES" ), QObject::tr( "Access time output raster" ) ) );
}

static bool load_terrain(void *self, const GeoReference *geo_ref, FFITerrain *output) {
  return static_cast<QgsPropagLoader*>(self)->load_terrain(geo_ref, output);
};

QVariantMap QgsPropagAlgorithm::processAlgorithm( const QVariantMap &parameters, QgsProcessingContext &context, QgsProcessingFeedback *feedback )
{
  const QString outputFile = parameterAsOutputLayer( parameters, QStringLiteral( "TIMES" ), context );
  QgsVectorLayer *ignitedElementsLayer = parameterAsVectorLayer( parameters, QStringLiteral( "IGNITED_ELEMENTS" ), context );
  if ( !ignitedElementsLayer )
    throw QgsProcessingException( invalidSourceError( parameters, QStringLiteral( "IGNITED_ELEMENTS" ) ) );

  const bool find_ref_change = parameterAsBoolean( parameters, QStringLiteral( "GENERATE_REFS" ), context );


  bool hasXValue = parameters.value( QStringLiteral( "CELL_SIZE_X" ) ).isValid();
  bool hasYValue = parameters.value( QStringLiteral( "CELL_SIZE_Y" ) ).isValid();
  double xSize, ySize;
  if ( ( hasXValue && !hasYValue ) || ( !hasXValue && hasYValue ) )
  {
    throw QgsProcessingException( QObject::tr( "Either set both X and Y cell size values or keep both as 'Not set'." ) );
  }
  else if ( hasXValue && hasYValue )
  {
    xSize = parameterAsDouble( parameters, QStringLiteral( "CELL_SIZE_X" ), context );
    ySize = parameterAsDouble( parameters, QStringLiteral( "CELL_SIZE_Y" ), context );
  } else {
    throw QgsProcessingException( QObject::tr( "Invalid CELL_SIZE" ) );
  }
  if (xSize <= 0) {
    throw QgsProcessingException( QObject::tr( "Invalid CELL_SIZE_X" ) );
  }
  if (ySize <= 0) {
    throw QgsProcessingException( QObject::tr( "Invalid CELL_SIZE_Y" ) );
  }

  QgsRectangle extent;
  if ( parameters.value( QStringLiteral( "EXTENT" ) ).isValid() )
  {
    extent = parameterAsExtent( parameters, QStringLiteral( "EXTENT" ), context );
  } else {
    throw QgsProcessingException( QObject::tr( "Invalid EXTENT" ) );
  }

  double max_time;
  if ( parameters.value( QStringLiteral( "MAX_SIMULATION_MINUTES" ) ).isValid() )
  {
    max_time = parameterAsDouble( parameters, QStringLiteral( "MAX_SIMULATION_MINUTES" ), context ) * 60.0;
  } else {
    throw QgsProcessingException( QObject::tr( "Invalid MAX_SIMULATION_MINUTES" ) );
  }
  if (max_time <= 0) {
    throw QgsProcessingException( QObject::tr( "MAX_SIMULATION_MINUTES must be > 0" ) );
  }

  GeoReference geo_ref;
  if (!GeoReference_south_up(extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum(), xSize, ySize, 9999, &geo_ref)) {
    throw QgsProcessingException( QObject::tr( "Could not create a valid GeoReference with given CELL_SIZE_? and EXTENT" ) );
  }

  Settings settings(geo_ref, max_time, find_ref_change);

  QByteArray outputFile_ba = outputFile.toLocal8Bit();
  const char *output_path = outputFile_ba.data();

  FFITimeFeature *initial_ignited_elements = NULL;
  size_t initial_ignited_elements_len = 0;
  QgsPropagLoader loader;
  FFITerrainLoader terrain_loader(&load_terrain, &loader);

  FFIPropagation propagation(settings, output_path, initial_ignited_elements, initial_ignited_elements_len, terrain_loader);

  FFIPropagation_run(propagation);

  QVariantMap outputs;
  outputs.insert( QStringLiteral( "TIMES" ), outputFile );
  return outputs;
}

///@endcond
