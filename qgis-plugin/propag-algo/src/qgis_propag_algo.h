#ifndef _QGIS_PROPAG_ALGO_H_
#define _QGIS_PROPAG_ALGO_H_

/* MSVC workarounds */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169163975144 // pi/2
#endif

#include "plugincontainer.h"
#include "qgisinterface.h"
#include "qgisplugin.h"
#include "qgsmapcanvas.h"
#include "qgsmessagelog.h"
#include "qgsvectorlayer.h"
#include <QAction>
#include <QApplication>
#include <iostream>
#include <qgsprocessingprovider.h>

class PropagAlgoPlugin : public QObject, public QgisPlugin {
  Q_OBJECT

public:
  /// @brief Constructor.
  /// @param qgis_if The Qgis interface.
  explicit PropagAlgoPlugin(QgisInterface *qgis_if);

  /// @brief Destructor
  virtual ~PropagAlgoPlugin() = default;

  /// @brief Called when the plugin is loaded.
  virtual void initGui() override;

  virtual void initProcessing();

  /// @brief Called when the plugin is unloaded.
  virtual void unload() override;

private:
  QgisInterface *m_qgis_if;
};

#endif
