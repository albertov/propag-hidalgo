#ifndef INTERFACE_H
#define INTERFACE_H

#include "qgsprocessingalgorithm.h"
#include <string>

class Base {
public:
  Base();
  virtual ~Base();
  virtual QgsProcessingAlgorithm *makeAlgorithm() = 0;
};

typedef Base *(*plugin_create_t)();
typedef void (*plugin_destroy_t)(Base *);

#endif // INTERFACE_H
