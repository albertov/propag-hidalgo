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

class FireSimpleCuda {
public:
  T speed_max;
  T azimuth_max;
  T eccentricity;

  __device__ FireSimpleCuda()
      : speed_max(0.0), azimuth_max(0.0), eccentricity(0.0) {}

  __device__ FireSimpleCuda(T speed_max, T azimuth_max, T eccentricity)
      : speed_max(speed_max), azimuth_max(azimuth_max),
        eccentricity(eccentricity) {};

  __device__ volatile FireSimpleCuda &
  operator=(const FireSimpleCuda &other) volatile {
    speed_max = other.speed_max;
    azimuth_max = other.azimuth_max;
    eccentricity = other.eccentricity;
    return *this;
  }
};
#define FireSimpleCuda_NULL FireSimpleCuda()

class PointRef {
public:
  float time;
  ushort2 pos;
  FireSimpleCuda fire;

  __device__ PointRef()
      : time(MAX_TIME), pos(make_ushort2(USHRT_MAX, USHRT_MAX)), fire() {};
  __device__ PointRef(float time, ushort2 pos, FireSimpleCuda fire)
      : time(time), pos(pos), fire(fire) {};
  __device__ PointRef(float time, int2 pos, FireSimpleCuda fire)
      : time(time), pos(make_ushort2(pos.x, pos.y)), fire(fire) {};

  __device__ volatile PointRef &operator=(const PointRef &other) volatile {
    pos.x = other.pos.x;
    pos.y = other.pos.y;
    fire = other.fire;
    // time must be written last
    __threadfence();
    time = other.time;
    return *this;
  }
};
#define PointRef_NULL PointRef()

typedef struct Settings {
  GeoReference geo_ref;
  float max_time;
} Settings;

class ALIGN Point {
public:
  float time;
  FireSimpleCuda fire;
  PointRef reference;

  __device__ Point() : time(MAX_TIME), fire(), reference() {};
  __device__ Point(float time, FireSimpleCuda fire, PointRef reference)
      : time(time), fire(fire), reference(reference) {};

  __device__ volatile Point &operator=(const Point &other) volatile {
    fire = other.fire;
    reference = other.reference;
    // time must be written last
    __threadfence();
    time = other.time;
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

__device__ inline bool similar_fires_(const volatile FireSimpleCuda &a,
                                      const volatile FireSimpleCuda &b) {
  return (abs(a.speed_max - b.speed_max) < 0.1 &&
          abs(a.azimuth_max - b.azimuth_max) < (5.0 * (2 * PI) / 360.0) &&
          abs(a.eccentricity - b.eccentricity) < 0.05);
}

__device__ inline int signum(int val) { return int(0 < val) - int(val < 0); }

class DDA {
  const int2 from_;
  const int2 to_;
  const int2 step_;
  int2 cur_;
  float2 tmax_;
  const float2 delta_;

public:
  __device__ DDA(uint2 from, uint2 to)
      : from_(make_int2(from.x, from.y)), to_(make_int2(to.x, to.y)),
        step_(make_int2(signum(to_.x - from_.x), signum(to_.y - from_.y))),
        cur_(from_), tmax_(make_float2(1.0 / float(abs(to_.x - from_.x)),
                                       1.0 / float(abs(to_.y - from_.y)))),
        delta_(tmax_) {};

  __device__ bool next(int2 &result) {
    result = cur_;
    if (result.x == to_.x && result.y == to_.y) {
      cur_.x += step_.x;
      cur_.y += step_.y;
      return true;
    }
    bool valid_x = (step_.x > 0 ? cur_.x <= to_.x : cur_.x >= to_.x);
    bool valid_y = (step_.y > 0 ? cur_.y <= to_.y : cur_.y >= to_.y);
    if (!(valid_x && valid_y))
      return false;
    if (abs(tmax_.x - tmax_.y) < 1e-6) {
      cur_.x += step_.x;
      cur_.y += step_.y;
      tmax_.x += delta_.x;
      tmax_.y += delta_.y;
    } else if (tmax_.x < tmax_.y) {
      cur_.x += step_.x;
      tmax_.x += delta_.x;
    } else {
      cur_.y += step_.y;
      tmax_.y += delta_.y;
    }
    return true;
  }
};

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
  float denom = (1.0 - from.fire.eccentricity * cos(angle));
  float speed;
  if (denom > 1e-6) {
    float factor = (1.0 - from.fire.eccentricity) / denom;
    speed = from.fire.speed_max * factor;
  } else {
    speed = from.fire.speed_max; // FIXME should be speed0
  }
  float dx = (from_pos.x - to_pos.x) * geo_ref.transform.gt.dx;
  float dy = (from_pos.y - to_pos.y) * geo_ref.transform.gt.dy;
  float distance = sqrt(dx * dx + dy * dy);
  return from.time + (distance / speed);
}

#endif
