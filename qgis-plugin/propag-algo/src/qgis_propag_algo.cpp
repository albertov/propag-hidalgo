#include "qgis_propag_algo.h"
#include "qgsproviderpropag.h"
#include <qgsprocessingprovider.h>
#include <qgsapplication.h>
#include <qgsmessagebar.h>
#include <qgsprocessingregistry.h>



namespace {
   const QString s_name = QStringLiteral("Wildfire propagator algorithm Plugin");
   const QString s_description = QStringLiteral("Sample Plugin");
   const QString s_category = QStringLiteral("Plugins");
   const QString s_version = QStringLiteral("Version 0.1");
   const QString s_icon =  QStringLiteral( ":/plugin.svg" );
   const QgisPlugin::PluginType s_type = QgisPlugin::UI;
}

QGISEXTERN QgisPlugin* classFactory(QgisInterface* qgis_if)
{
   std::cout << "::classFactory" << std::endl;
   return new PropagAlgoPlugin(qgis_if);
}

// See QGIS breaking change introduced qith QGIS 3.22:
// https://github.com/qgis/QGIS/commit/b3c5cf8d5fc1fdc289f1449df548acf9268140c6
#if _QGIS_VERSION_INT >= 32200
// Receny versions of QGIS use pointer to strings to pass the plugin information.
QGISEXTERN const QString* name() {
   return &s_name;
}

QGISEXTERN const QString* description() {
   return &s_description;
}

QGISEXTERN const QString* category() {
   return &s_category;
}

QGISEXTERN const QString* version() {
   return &s_version;
}

QGISEXTERN const QString* icon() {
   return &s_icon;
}
#else
// Older versions of QGIS return the plugin names as copies.
QGISEXTERN const QString name() {
   return s_name;
}

QGISEXTERN const QString description() {
   return s_description;
}

QGISEXTERN const QString category() {
   return s_category;
}

QGISEXTERN const QString version() {
   return s_version;
}

QGISEXTERN const QString icon() {
   return s_icon;
}
#endif

QGISEXTERN int type() {
   return s_type;
}

QGISEXTERN void unload(QgisPlugin* plugin) {
   std::cout << "::unload" << std::endl;
   //delete plugin;
}

PropagAlgoPlugin::PropagAlgoPlugin(QgisInterface* iface) : QgisPlugin(s_name, s_description, s_category, s_version, s_type), m_qgis_if(iface) {
}


void PropagAlgoPlugin::unload() {
   // TODO - need to remove the actions from the menu again.
}

void PropagAlgoPlugin::initGui() {
   std::cout << "PropagAlgoPlugin::initGui" << std::endl;
   initProcessing();

}


void PropagAlgoPlugin::initProcessing()
{

    QgsProcessingProvider *processing_provider = new PropagProvider(QgsApplication::processingRegistry());
    bool ok = QgsApplication::processingRegistry()->addProvider(processing_provider);
    std::cout << "PropagAlgoPlugin::initProcessing" << std::endl;
    if(!ok) {
      m_qgis_if->messageBar()->pushMessage("Could not load propag processing plugin", Qgis::Critical);
    }
}
