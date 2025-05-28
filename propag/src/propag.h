#ifndef PROPAG_H
#define PROPAG_H

#include "firelib_cuda.h"
#include <cooperative_groups.h>
#include <float.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ALIGN __align__(16)
#define ASSERT(X) assert(X)

class ALIGN FireSimpleCuda {
public:
  T speed_max;
  T azimuth_max;
  T eccentricity;

  __device__ FireSimpleCuda()
      : speed_max(0.0), azimuth_max(0.0), eccentricity(0.0) {}

  __device__ FireSimpleCuda(const T speed_max, const T azimuth_max, const T eccentricity)
      : speed_max(speed_max), azimuth_max(azimuth_max),
        eccentricity(eccentricity) {};

  __device__ volatile FireSimpleCuda &
  operator=(volatile FireSimpleCuda &other) volatile {
    speed_max = other.speed_max;
    azimuth_max = other.azimuth_max;
    eccentricity = other.eccentricity;
    return *this;
  }
  __device__ FireSimpleCuda &
  operator=(const FireSimpleCuda &other) {
    speed_max = other.speed_max;
    azimuth_max = other.azimuth_max;
    eccentricity = other.eccentricity;
    return *this;
  }

  __device__ FireSimpleCuda(volatile FireSimpleCuda *other)
      : speed_max(other->speed_max), azimuth_max(other->azimuth_max),
        eccentricity(other->eccentricity) {};

  __device__ FireSimpleCuda(volatile FireSimpleCuda &other)
      : speed_max(other.speed_max), azimuth_max(other.azimuth_max),
        eccentricity(other.eccentricity) {};

  __device__ FireSimpleCuda(const FireSimpleCuda &other)
      : speed_max(other.speed_max), azimuth_max(other.azimuth_max),
        eccentricity(other.eccentricity) {};
};
#define FireSimpleCuda_NULL FireSimpleCuda()

class ALIGN PointRef {
public:
  float time;
  ushort2 pos;
  FireSimpleCuda fire;

  __device__ PointRef()
      : time(MAX_TIME), pos(make_ushort2(USHRT_MAX, USHRT_MAX)), fire() {};
  __device__ PointRef(const float time, const ushort2 pos, const FireSimpleCuda fire)
      : time(time), pos(pos), fire(fire) {};
  __device__ PointRef(const float time, const int2 pos, const FireSimpleCuda fire)
      : time(time), pos(make_ushort2(pos.x, pos.y)), fire(fire) {};
  __device__ PointRef(volatile PointRef &other)
      : time(other.time), pos(make_ushort2(other.pos.x, other.pos.y)),
        fire(other.fire) {};
  __device__ PointRef(const PointRef &other)
      : time(other.time), pos(make_ushort2(other.pos.x, other.pos.y)),
        fire(other.fire) {};
  __device__ PointRef(volatile PointRef *other)
      : time(other->time), pos(make_ushort2(other->pos.x, other->pos.y)),
        fire(other->fire) {};

  __device__ volatile PointRef &operator=(volatile PointRef &other) volatile {
    time = other.time;
    pos.x = other.pos.x;
    pos.y = other.pos.y;
    fire = other.fire;
    return *this;
  }
  __device__ const PointRef &operator=(const PointRef &other) {
    time = other.time;
    pos.x = other.pos.x;
    pos.y = other.pos.y;
    fire = other.fire;
    return *this;
  }
};
#define PointRef_NULL PointRef()

typedef struct ALIGN Settings {
  GeoReference geo_ref;
  float max_time;
} Settings;

class ALIGN Point {
public:
  float time;
  FireSimpleCuda fire;
  PointRef reference;

  __device__ Point() : time(MAX_TIME), fire(), reference() {};
  __device__ Point(float time, const FireSimpleCuda fire, const PointRef reference)
      : time(time), fire(fire), reference(reference) {};

  __device__ Point(volatile Point &other)
      : time(other.time), fire(other.fire), reference(other.reference) {};

  __device__ Point(volatile Point *other)
      : time(other->time), fire(other->fire), reference(other->reference) {};


  /*
  __device__ volatile Point &operator=(const Point &other) volatile {
    time = other.time;
    fire = other.fire;
    reference = other.reference;
    return *this;
  }
  */
  __device__ volatile Point operator=(volatile Point other) volatile {
    time = other.time;
    fire = other.fire;
    reference = other.reference;
    return *this;
  }
};

#define Point_NULL Point()

__device__ inline uint2 index_2d(uint2 gridIx) {
  return make_uint2(threadIdx.x + blockIdx.x * blockDim.x +
                        gridIx.x * gridDim.x * blockDim.x,
                    threadIdx.y + blockIdx.y * blockDim.y +
                        gridIx.y * gridDim.y * blockDim.y);
}

__device__ inline FireSimpleCuda load_fire(size_t idx, const float *speed_max,
                                           const float *azimuth_max,
                                           const float *eccentricity) {
  return FireSimpleCuda(speed_max[idx], azimuth_max[idx], eccentricity[idx]);
}

__device__ inline bool is_point_null(const Point &p) {
  return !(p.time < MAX_TIME);
}

__device__ inline bool is_fire_null(const volatile FireSimpleCuda &f) {
  return f.speed_max == 0.0;
}

__device__ inline volatile Point
load_point(const GeoReference &geo_ref, uint2 pos, size_t idx,
           const float *speed_max, const float *azimuth_max,
           const float *eccentricity, volatile float *time,
           volatile unsigned short *ref_x, volatile unsigned short *ref_y) {
  float p_time = time[idx];
  const FireSimpleCuda fire = load_fire(idx, speed_max, azimuth_max, eccentricity);
  const PointRef ref;

  if (p_time < MAX_TIME) {
    ushort2 ref_pos = make_ushort2(ref_x[idx], ref_y[idx]);
    ASSERT(!(ref_pos.x == USHRT_MAX || ref_pos.y == USHRT_MAX));
    ASSERT(ref_pos.x < geo_ref.width || ref_pos.y < geo_ref.height);
    if (ref_pos.x == pos.x && ref_pos.y == pos.y) {
      return Point(p_time, fire, PointRef(p_time, ref_pos, fire));
    }
    size_t ref_ix = ref_pos.x + ref_pos.y * geo_ref.width;
    float ref_time = time[ref_ix];
    ASSERT(ref_time != MAX_TIME);
    const FireSimpleCuda ref_fire = load_fire(ref_ix, speed_max, azimuth_max, eccentricity);
    ASSERT(!is_fire_null(ref_fire));
    return Point(p_time, fire, PointRef(ref_time, ref_pos, ref_fire));
  };
  return Point_NULL;
}

__device__ inline bool similar_fires_(const volatile FireSimpleCuda &a,
                                      const volatile FireSimpleCuda &b) {
  return (abs(a.speed_max - b.speed_max) < 0.1 &&
          abs(a.azimuth_max - b.azimuth_max) < (5.0 * (2 * PI) / 360.0) &&
          abs(a.eccentricity - b.eccentricity) < 0.05);
}

__device__ inline int signum(int val) { return int(0 < val) - int(val < 0); }

__device__ inline int2 neighbor_direction(uint2 ufrom, uint2 uto) {
  int2 from = make_int2(ufrom.x, ufrom.y);
  int2 to = make_int2(uto.x, uto.y);
  if (from.x == to.x && from.y == to.y) {
    return make_int2(0.0, 0.0);
  }
  int2 step = make_int2(signum(to.x - from.x), signum(to.y - from.y));
  int2 tmax = make_int2(abs(to.x - from.x), abs(to.y - from.y));

  if (tmax.x == tmax.y) {
    return step;
  } else if (tmax.x > tmax.y) {
    return make_int2(step.x, 0);
  } else {
    return make_int2(0, step.y);
  }
}

__device__ inline float time_to(const GeoReference &geo_ref,
                                const PointRef &from, const uint2 to) {

  int2 from_pos = make_int2(from.pos.x, from.pos.y);
  int2 to_pos = make_int2(to.x, to.y);
  float azimuth;
  if (geo_ref.transform.gt.dy < 0) {
    // north-up geotransform
    azimuth = atan2f(to_pos.x - from_pos.x, from_pos.y - to_pos.y);
  } else {
    azimuth = atan2f(to_pos.x - from_pos.x, to_pos.y - from_pos.y);
    // south-up geotransform
  };
  float angle = abs(azimuth - from.fire.azimuth_max);
  float factor = (1.0 - from.fire.eccentricity) /
                 (1.0 - from.fire.eccentricity * cos(angle));

  float dx = (from.pos.x - to.x) * geo_ref.transform.gt.dx;
  float dy = (from.pos.y - to.y) * geo_ref.transform.gt.dy;
  float distance = sqrt(dx * dx + dy * dy);
  ASSERT(distance>0);

  float speed = from.fire.speed_max * factor;
  if (speed > 1e-6) {
    return from.time + (distance / speed);
  } else {
    return MAX_TIME;
  }
}

#endif
