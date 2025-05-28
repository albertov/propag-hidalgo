#ifndef QGSPROVIDERPROPAG_H
#define QGSPROVIDERPROPAG_H

#include <qgsapplication.h>
#include <qgsprocessingprovider.h>

class PropagProvider : public QgsProcessingProvider {
  Q_OBJECT

public:
  PropagProvider(QObject *parent = nullptr);

  QIcon icon() const override;
  QString svgIconPath() const override;
  QString id() const override;
  QString helpId() const override;
  QString name() const override;
  bool supportsNonFileBasedOutput() const override;

protected:
  void loadAlgorithms() override;
};

#endif
