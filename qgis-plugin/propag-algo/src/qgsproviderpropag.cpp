#include "qgsproviderpropag.h"
#include "qgsalgorithmpropag.h"

PropagProvider::PropagProvider(QObject *parent, QgsProcessingAlgorithm *algo)
    : QgsProcessingProvider(parent), algo(algo) {}

QIcon PropagProvider::icon() const {

  return QgsApplication::getThemeIcon(
      QStringLiteral(":/plugins/propag25/flame.svg"));
}

QString PropagProvider::svgIconPath() const {

  return QgsApplication::iconPath(
      QStringLiteral(":/plugins/propag25/flame.svg"));
}

QString PropagProvider::id() const { return QStringLiteral("propagprovider"); }

QString PropagProvider::name() const { return tr("Wildfire Simulator"); }

QString PropagProvider::helpId() const {
  return tr("A cellular automata wildfile siumlator");
}

bool PropagProvider::supportsNonFileBasedOutput() const { return true; }

void PropagProvider::loadAlgorithms() {
  // const QgsScopedRuntimeProfile profile(QObject::tr("PropagProvider"));
  std::cout << "loadAlgorithms" << std::endl;
  addAlgorithm(algo);
}
