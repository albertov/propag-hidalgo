#include <stdio.h>
#include "geometry.h"
#include "firelib_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void propag(
    Settings settings,
    const float *speed_max,
    size_t speed_max_len,
    const float *azimuth_max,
    long azimuth_max_len,
    const float *eccentricity,
    size_t eccentricity_len,
    float *time,
    size_t *refs_x,
    size_t *refs_y,
    unsigned *progress
  )
{
  extern __shared__ Point shared[];
  printf("Hello World from GPU %d %d!\n", settings.geo_ref.width, azimuth_max_len);
}

#ifdef __cplusplus
}
#endif
