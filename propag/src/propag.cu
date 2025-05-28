#include "firelib_cuda.h"
#include <float.h>
#include <stdio.h>

#define MAX_TIME FLT_MAX
#define PRELOAD_POINT(X, Y)                                                    \
  preload_point(settings.geo_ref, make_int2((X), (Y)), speed_max, azimuth_max, \
                eccentricity, time, refs_x, refs_y, shared)

__device__ inline uint2 index_2d() {
  return make_uint2(threadIdx.x + blockIdx.x * blockDim.x,
                    threadIdx.y + blockIdx.y * blockDim.y);
}

__device__ inline size_t compute_shared_ix(uint2 pos) {
  size_t shared_x = pos.x % blockDim.x;
  size_t shared_y = pos.y % blockDim.x;
  size_t ix = shared_x + shared_y * (blockDim.x + HALO_RADIUS * 2);
  assert(ix < (blockDim.x + HALO_RADIUS * 2) * (blockDim.y + HALO_RADIUS * 2));
  assert(shared_x < blockDim.x + HALO_RADIUS * 2);
  assert(shared_y < blockDim.y + HALO_RADIUS * 2);
  return ix;
}

__device__ inline bool offset_shared_ix(const GeoReference &geo_ref,
                                        int2 offset, size_t *result) {
  uint2 upos = index_2d();
  int2 pos = make_int2(upos.x + offset.x, upos.y + offset.y);
  if (pos.x >= 0 && pos.y >= 0 && pos.x < geo_ref.width &&
      pos.y < geo_ref.height) {
    *result = compute_shared_ix(make_uint2(pos.x, pos.y));
    return true;
  } else {
    return false;
  }
}

__device__ inline bool offset_ix(const GeoReference &geo_ref, int2 offset,
                                 size_t *result) {
  uint2 upos = index_2d();
  int2 pos = make_int2(upos.x + offset.x, upos.y + offset.y);
  if (pos.x >= 0 && pos.y >= 0 && pos.x < geo_ref.width &&
      pos.y < geo_ref.height) {
    *result = pos.x + pos.y * geo_ref.width;
    return true;
  } else {
    return false;
  }
}

__device__ inline FireSimpleCuda load_fire(size_t idx, const float *speed_max,
                                           const float *azimuth_max,
                                           const float *eccentricity) {
  FireSimpleCuda ret = FireSimpleCuda_NULL;
  ret.speed_max = speed_max[idx];
  ret.azimuth_max = azimuth_max[idx];
  ret.eccentricity = eccentricity[idx];
  return ret;
}

__device__ inline Point load_point(const GeoReference &geo_ref, uint2 pos,
                                   size_t idx, const float *speed_max,
                                   const float *azimuth_max,
                                   const float *eccentricity,
                                   volatile float *time, volatile size_t *ref_x,
                                   volatile size_t *ref_y) {
  float p_time = time[idx];
  if (p_time == MAX_TIME) {
    return Point_NULL;
  }
  FireSimpleCuda fire = load_fire(idx, speed_max, azimuth_max, eccentricity);
  uint2 ref_pos = make_uint2(ref_x[idx], ref_y[idx]);
  assert(!(ref_pos.x == SIZE_MAX || ref_pos.y == SIZE_MAX));
  size_t ref_ix = ref_pos.x + ref_pos.y * geo_ref.width;
  FireSimpleCuda ref_fire =
      load_fire(ref_ix, speed_max, azimuth_max, eccentricity);
  float ref_time = time[ref_ix];
  assert(ref_time != MAX_TIME);

  PointRef ref = PointRef_NULL;
  ref.time = ref_time;
  ref.pos_x = ref_pos.x;
  ref.pos_y = ref_pos.y;
  ref.fire = ref_fire;

  Point result = Point_NULL;
  result.time = p_time;
  result.fire = fire;
  result.reference = ref;

  return result;
}

__device__ inline Point
preload_point(const GeoReference &geo_ref, int2 offset, const float *speed_max,
              const float *azimuth_max, const float *eccentricity,
              volatile float *time, volatile size_t *ref_x,
              volatile size_t *ref_y, Point *shared) {
  size_t shared_off;
  size_t global_off;
  uint2 idx_2d = index_2d();
  uint2 point_pos = make_uint2(idx_2d.x + offset.x, idx_2d.y + offset.y);
  bool shared_good = offset_shared_ix(geo_ref, offset, &shared_off);
  bool global_good = offset_ix(geo_ref, offset, &global_off);
  if (global_good && shared_good) {
    Point p = load_point(geo_ref, point_pos, global_off, speed_max, azimuth_max,
                         eccentricity, time, ref_x, ref_y);
    shared[shared_off] = p;
    return p;
  } else if (shared_good) {
    shared[shared_off] = Point_NULL;
    ;
    return Point_NULL;
  } else if (global_good) {
    assert(false);
    return Point_NULL;
  } else {
    return Point_NULL;
  }
}

__device__ inline bool is_point_null(const Point &p) {
  return p.time == MAX_TIME;
}

__device__ inline bool is_fire_null(const FireSimpleCuda &f) {
  return f.speed_max == 0.0 && f.azimuth_max == 0.0 && f.eccentricity == 0.0;
}

__device__ inline bool similar_fires(const FireSimpleCuda &a,
                                     const FireSimpleCuda &b) {
  return (abs(a.speed_max - b.speed_max) < 1.0 &&
          abs(a.azimuth_max - b.azimuth_max) < (5.0 / 2 * PI) &&
          abs(a.eccentricity - b.eccentricity) < 0.1);
}

__device__ inline int signum(int val) { return (0 < val) - (val < 0); }

__device__ inline uint2 neighbor_in_direction(uint2 ufrom, uint2 uto) {
  int2 from = make_int2(ufrom.x, ufrom.y);
  int2 to = make_int2(uto.x, uto.y);
  if (from.x == to.x && from.y == to.y) {
    return make_uint2(from.x, from.y);
  }
  int2 step = make_int2(signum(to.x - from.x), signum(to.y - from.y));
  int2 tmax = make_int2(abs(to.x - from.x), abs(to.y - from.y));

  if (tmax.x == tmax.y) {
    return make_uint2(from.x + step.x, from.y + step.y);
  } else if (tmax.x > tmax.y) {
    return make_uint2(from.x + step.x, from.y);
  } else {
    return make_uint2(from.x, from.y + step.y);
  }
}
__device__ inline float time_to(const GeoReference &geo_ref,
                                const PointRef &from, const uint2 to) {

  int2 from_pos = make_int2(from.pos_x, from.pos_y);
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

  float dx = (from.pos_x - to.x) * geo_ref.transform.gt.dx;
  float dy = (from.pos_y - to.y) * geo_ref.transform.gt.dy;
  float distance = sqrt(dx * dx + dy * dy);

  float speed = from.fire.speed_max * factor;
  return from.time + (distance / speed);
}

#ifdef __cplusplus
extern "C" {
#endif

__global__ void propag(const Settings &settings, const float *speed_max,
                       const float *azimuth_max, const float *eccentricity,
                       float volatile *time, size_t volatile *refs_x,
                       size_t volatile *refs_y, unsigned volatile *progress) {
  extern __shared__ Point shared[];

  uint2 idx_2d = index_2d();

  size_t global_ix = idx_2d.x + idx_2d.y * settings.geo_ref.width;
  bool in_bounds =
      idx_2d.x < settings.geo_ref.width && idx_2d.y < settings.geo_ref.height;

  size_t block_ix = blockIdx.x + blockIdx.y * gridDim.x;
  assert(block_ix < gridDim.x * gridDim.y);

  // Initialize progress to no-progress
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    progress[block_ix] = 0;
  }

  //////////////////////////////////////////////////////
  // First phase, load data from global to shared memory
  //////////////////////////////////////////////////////
  FireSimpleCuda fire = FireSimpleCuda_NULL;
  bool is_new = false;
  if (in_bounds) {
    Point point = PRELOAD_POINT(0, 0);
    is_new = is_point_null(point);
    if (is_new) {
      fire = load_fire(global_ix, speed_max, azimuth_max, eccentricity);
    };
    bool x_near_border = threadIdx.x < HALO_RADIUS;
    bool y_near_border = threadIdx.y < HALO_RADIUS;
    if (x_near_border) {
      PRELOAD_POINT(-HALO_RADIUS, 0);
      PRELOAD_POINT(blockDim.x, 0);
    };
    if (y_near_border) {
      PRELOAD_POINT(0, -HALO_RADIUS);
      PRELOAD_POINT(0, blockDim.y);
    };
    if (x_near_border && y_near_border) {
      PRELOAD_POINT(-HALO_RADIUS, -HALO_RADIUS);
      PRELOAD_POINT(blockDim.x, blockDim.y);
    };
  }; // end in_bounds
  __syncthreads();

  ////////////////////////////////////////////////////////////////////////
  // Begin neighbor analysys
  ////////////////////////////////////////////////////////////////////////

  Point best = Point_NULL;
  unsigned width = settings.geo_ref.width;
  unsigned height = settings.geo_ref.height;

  if (in_bounds && is_new) {
#pragma unroll
    for (int i = -1; i < 2; i++) {
#pragma unroll
      for (int j = -1; j < 2; j++) {
        // Skip self and out-of-bounds neighbors
        if (i == 0 && j == 0)
          continue;
        int2 pos = make_int2(idx_2d.x + i, idx_2d.y + j);
        if (!(pos.x > 0 && pos.y > 0 && pos.x < width && pos.y < height))
          continue;

        // Good neighbor
        uint2 neighbor_pos = make_uint2(pos.x, pos.y);
        Point neighbor_point = shared[compute_shared_ix(neighbor_pos)];

        // Check if neighbor's reference is usable
        PointRef reference = neighbor_point.reference;
        if (reference.pos_x = pos.x && reference.pos_y == pos.y) {
          reference = PointRef_NULL;
        } else {
          uint2 possible_blockage_pos = neighbor_in_direction(
              idx_2d, make_uint2(reference.pos_x, reference.pos_y));
          Point possible_blockage =
              shared[compute_shared_ix(possible_blockage_pos)];
          if (!(!is_point_null(possible_blockage) &&
                !is_fire_null(possible_blockage.fire) &&
                similar_fires(possible_blockage.fire, reference.fire))) {
            reference = PointRef_NULL;
          };
        };

        // Look for a new candidate for best time
        Point candidate = Point_NULL;
        if (!is_fire_null(fire) && reference.time != MAX_TIME) {
          // We are combustible
          if (!similar_fires(fire, reference.fire)) {
            // we can't reuse reference because fire is different to ours.
            // Try to use neighbor as reference
            PointRef r = neighbor_point.reference;
            if (r.pos_x != idx_2d.x && r.pos_y != idx_2d.y &&
                similar_fires(fire, r.fire)) {
              reference = r;
            };
          };
          if (reference.time != MAX_TIME) {
            float time = time_to(settings.geo_ref, reference, idx_2d);
            if (time < settings.max_time) {
              candidate.time = time;
              candidate.fire = fire;
              candidate.reference = reference;
            }
          };
        } else if (reference.time != MAX_TIME) {
          // We are not combustible but reference can be used.
          // We assign an access time but a None fire
          float time = time_to(settings.geo_ref, reference, idx_2d);
          if (time < settings.max_time) {
            candidate.time = time;
            candidate.fire = FireSimpleCuda_NULL;
            candidate.reference = reference;
          }
        }

        // If no candidate or candidate improves use it as best
        if (is_point_null(best) || candidate.time < best.time) {
          best = candidate;
        }
      };
    };
  };

  ///////////////////////////////////////////////////
  // End of neighbor analysys, save point if improves
  ///////////////////////////////////////////////////
  bool improved = false;
  if (in_bounds && best.time < MAX_TIME) {
    shared[compute_shared_ix(idx_2d)] = best;
    // FIXME
    // time[global_ix] = best.time;
    // refs_x[global_ix] = best.reference.pos_x;
    // refs_y[global_ix] = best.reference.pos_y;
    // improved = true;
  };

  ///////////////////////////////////////////////////
  // Wait for other threads to end their analysis and
  // check if any has improved. Then if we're the first
  // thread of the block mark progress
  ///////////////////////////////////////////////////
  if (__syncthreads_or(improved) && threadIdx.x == 0 && threadIdx.y == 0) {
    size_t block_ix = blockIdx.x + blockIdx.y * gridDim.x;
    progress[block_ix] = 1;
  };
}

#ifdef __cplusplus
}
#endif
