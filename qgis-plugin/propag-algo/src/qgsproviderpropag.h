#ifndef QGSPROVIDERPROPAG_H
#define QGSPROVIDERPROPAG_H

#include "plugincontainer.h"
#include <qgsapplication.h>
#include <qgsprocessingprovider.h>

class PropagProvider : public QgsProcessingProvider {
  Q_OBJECT

public:
  PropagProvider(QObject *parent = nullptr,
                 QgsProcessingAlgorithm *algo = nullptr);

  QIcon icon() const override;
  QString svgIconPath() const override;
  QString id() const override;
  QString helpId() const override;
  QString name() const override;
  bool supportsNonFileBasedOutput() const override;

protected:
  void loadAlgorithms() override;
  QgsProcessingAlgorithm *algo;
};

static inline PluginContainer *load_plugin(const std::string &path) {
  return new PluginContainer(path);
}
#endif
