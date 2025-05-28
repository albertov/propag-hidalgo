#ifndef PROPAG_H
#define PROPAG_H

#include "propag_host.h"
#include <cooperative_groups.h>
#include <float.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ALIGN
#define ASSERT(X) assert(X)

__device__ static inline void set_fire(FireSimpleCuda volatile *me,
                                       const FireSimpleCuda &other) {
  me->speed_max = other.speed_max;
  me->azimuth_max = other.azimuth_max;
  me->eccentricity = other.eccentricity;
}

__device__ static inline FireSimpleCuda load_fire(size_t idx,
                                                  const float *speed_max,
                                                  const float *azimuth_max,
                                                  const float *eccentricity) {
  return {.speed_max = speed_max[idx],
          .azimuth_max = azimuth_max[idx],
          .eccentricity = eccentricity[idx]};
}

__device__ static inline bool fire_is_null(const FireSimpleCuda &me) {
  return me.speed_max == 0.0;
}

class PointRef {
public:
  float time;
  ushort2 pos;

  __device__ PointRef()
      : time(FLT_MAX), pos(make_ushort2(USHRT_MAX, USHRT_MAX)) {};
  __device__ PointRef(float time, ushort2 pos) : time(time), pos(pos) {};
  __device__ PointRef(float time, int2 pos)
      : time(time), pos(make_ushort2(pos.x, pos.y)) {};

  __device__ volatile PointRef &operator=(const PointRef &other) volatile {
    pos.x = other.pos.x;
    pos.y = other.pos.y;
    time = other.time;
    return *this;
  }

  __device__ inline bool is_valid(const GeoReference &geo_ref) const {
    return !(pos.x == USHRT_MAX || pos.y == USHRT_MAX ||
             pos.x >= geo_ref.width || pos.y >= geo_ref.height);
  }
};
#define PointRef_NULL PointRef()

class ALIGN Point {
public:
  float time;
  FireSimpleCuda fire;
  PointRef reference;

  __device__ Point() : time(FLT_MAX), fire(), reference() {};
  __device__ Point(float time, FireSimpleCuda fire, PointRef reference)
      : time(time), fire(fire), reference(reference) {};

  __device__ volatile Point &operator=(const Point &other) volatile {
    set_fire(&fire, other.fire);
    reference = other.reference;
    time = other.time;
    return *this;
  }
  __device__ inline bool is_null() const { return !(time < FLT_MAX); }
};

#define Point_NULL Point()

__device__ inline uint2 index_2d(uint2 gridIx) {
  return make_uint2(threadIdx.x + blockIdx.x * blockDim.x +
                        gridIx.x * gridDim.x * blockDim.x,
                    threadIdx.y + blockIdx.y * blockDim.y +
                        gridIx.y * gridDim.y * blockDim.y);
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
  __device__ inline DDA(uint2 from, uint2 to)
      : from_(make_int2(from.x, from.y)), to_(make_int2(to.x, to.y)),
        step_(make_int2(signum(to_.x - from_.x), signum(to_.y - from_.y))),
        cur_(from_), tmax_(make_float2(1.0 / float(abs(to_.x - from_.x)),
                                       1.0 / float(abs(to_.y - from_.y)))),
        delta_(tmax_){};

  __device__ inline bool next(int2 &result) {
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

#endif
