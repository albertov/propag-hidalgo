#ifndef QGSALGORITHMPROPAG_H
#define QGSALGORITHMPROPAG_H

#define SIP_NO_FILE

#include "propag_host.h"
#include "qgis_sip.h"
#include "qgsapplication.h"
#include "qgsprocessingalgorithm.h"

///@cond PRIVATE

/**
 * Native align single raster algorithm for use in the modeler.
 */
class QgsPropagAlgorithm : public QgsProcessingAlgorithm {
public:
  QgsPropagAlgorithm() = default;
  void initAlgorithm(const QVariantMap &configuration = QVariantMap()) override;
  // Qgis::ProcessingAlgorithmFlags flags() const override;
  QString name() const override;
  QString displayName() const override;
  QStringList tags() const override;
  QString group() const override;
  QString groupId() const override;
  QString shortHelpString() const override;
  QgsPropagAlgorithm *createInstance() const override SIP_FACTORY;

protected:
  QVariantMap processAlgorithm(const QVariantMap &parameters,
                               QgsProcessingContext &context,
                               QgsProcessingFeedback *feedback) override;
};

#endif // QGSALGORITHMPROPAG_H
