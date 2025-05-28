#ifndef PROPAG_H
#define PROPAG_H

#include "firelib_cuda.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define ALIGN __align__(16)

class ALIGN FireSimpleCuda {
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

class ALIGN PointRef {
public:
  float time;
  uint2 pos;
  FireSimpleCuda fire;

  __device__ PointRef()
      : time(MAX_TIME), pos(make_uint2(SIZE_MAX, SIZE_MAX)), fire() {};
  __device__ PointRef(float time, uint2 pos, FireSimpleCuda fire)
      : time(time), pos(pos), fire(fire) {};
  __device__ PointRef(float time, int2 pos, FireSimpleCuda fire)
      : time(time), pos(make_uint2(pos.x, pos.y)), fire(fire) {};

  __device__ volatile PointRef &operator=(const PointRef &other) volatile {
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
  __device__ Point(float time, FireSimpleCuda fire, PointRef reference)
      : time(time), fire(fire), reference(reference) {};

  __device__ volatile Point &operator=(const Point &other) volatile {
    time = other.time;
    fire = other.fire;
    reference = other.reference;
    return *this;
  }
};

#define Point_NULL Point()

#endif
