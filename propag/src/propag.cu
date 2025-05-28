#include "propag.h"
#include <float.h>
#include <stdio.h>

__device__ inline uint2 index_2d() {
  return make_uint2(threadIdx.x + blockIdx.x * blockDim.x,
                    threadIdx.y + blockIdx.y * blockDim.y);
}

__device__ inline FireSimpleCuda load_fire(size_t idx, const float *speed_max,
                                           const float *azimuth_max,
                                           const float *eccentricity) {
  return FireSimpleCuda(speed_max[idx], azimuth_max[idx], eccentricity[idx]);
}

__device__ inline Point load_point(const GeoReference &geo_ref, uint2 pos,
                                   size_t idx, const float *speed_max,
                                   const float *azimuth_max,
                                   const float *eccentricity,
                                   volatile float *time, volatile size_t *ref_x,
                                   volatile size_t *ref_y) {
  Point result =
      Point(time[idx], load_fire(idx, speed_max, azimuth_max, eccentricity),
            PointRef());

  if (result.time < MAX_TIME) {
    result.reference.pos = make_uint2(ref_x[idx], ref_y[idx]);
    assert(!(result.reference.pos.x == SIZE_MAX ||
             result.reference.pos.y == SIZE_MAX));
    assert(result.reference.pos.x < geo_ref.width ||
           result.reference.pos.y < geo_ref.height);
    size_t ref_ix =
        result.reference.pos.x + result.reference.pos.y * geo_ref.width;

    if (result.reference.pos.x == pos.x && result.reference.pos.y == pos.y) {
      result.reference.fire = result.fire;
      result.reference.time = result.time;
    } else {
      result.reference.time = time[ref_ix];
      result.reference.fire =
          load_fire(ref_ix, speed_max, azimuth_max, eccentricity);
    };
    assert(result.reference.time != MAX_TIME);
  };
  return result;
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

  float speed = from.fire.speed_max * factor;
  return from.time + (distance / speed);
}

//////////////////////////////////////////////////////////////////////////////
/// propag
//////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

__global__ void propag(const Settings &settings, const float *speed_max,
                       const float *azimuth_max, const float *eccentricity,
                       float volatile *time, size_t volatile *refs_x,
                       size_t volatile *refs_y, unsigned volatile *progress) {
  extern __shared__ Point shared[];

  unsigned width = settings.geo_ref.width;
  unsigned height = settings.geo_ref.height;

  uint2 idx_2d = index_2d();

  size_t global_ix = idx_2d.x + idx_2d.y * width;
  bool in_bounds = idx_2d.x < width && idx_2d.y < height;

  size_t block_ix = blockIdx.x + blockIdx.y * gridDim.x;
  assert(block_ix < gridDim.x * gridDim.y);

  // Initialize progress to no-progress
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    progress[block_ix] = 0;
  };

  // Calculate global indices
  int global_x = idx_2d.x;
  int global_y = idx_2d.y;

  // Calculate local (shared memory) indices
  int local_x = threadIdx.x + HALO_RADIUS;
  int local_y = threadIdx.y + HALO_RADIUS;
  int shared_width = blockDim.x + HALO_RADIUS * 2;
  int local_ix = local_x + local_y * shared_width;

  bool any_improved = false;
  int num_iterations = 0;
  do {
    //////////////////////////////////////////////////////
    // First phase, load data from global to shared memory
    //////////////////////////////////////////////////////
    if (in_bounds) {
#define LOAD(LOCAL_X, LOCAL_Y, GLOBAL_X, GLOBAL_Y)                             \
  if ((GLOBAL_X) >= 0 && (GLOBAL_X) < width && (GLOBAL_Y) >= 0 &&              \
      (GLOBAL_Y) < height) {                                                   \
    shared[(LOCAL_X) + (LOCAL_Y) * shared_width] =                             \
        load_point(settings.geo_ref, make_uint2(GLOBAL_X, GLOBAL_Y),           \
                   (GLOBAL_X) + (GLOBAL_Y) * width, speed_max, azimuth_max,    \
                   eccentricity, time, refs_x, refs_y);                        \
  } else {                                                                     \
    shared[(LOCAL_X) + (LOCAL_Y) * shared_width] = Point_NULL;                 \
  }
      // Load the central block data
      if (num_iterations == 0) {
        LOAD(local_x, local_y, global_x, global_y);
        num_iterations++;
      }

      // Load the halo regions
      bool x_near_x0 = threadIdx.x < HALO_RADIUS;
      bool y_near_y0 = threadIdx.y < HALO_RADIUS;
      if (y_near_y0) {
        // Top halo
        LOAD(local_x, local_y - HALO_RADIUS, global_x, global_y - HALO_RADIUS);
        // Bottom halo
        LOAD(local_x, local_y + blockDim.y, global_x, global_y + blockDim.y);
      }
      if (x_near_x0) {
        // Left halo
        LOAD(local_x - HALO_RADIUS, local_y, global_x - HALO_RADIUS, global_y);
        // Right halo
        LOAD(local_x + blockDim.x, local_y, global_x + blockDim.x, global_y);
      }
      if (x_near_x0 && y_near_y0) {
        // corners
        LOAD(local_x - HALO_RADIUS, local_y - HALO_RADIUS,
             global_x - HALO_RADIUS, global_y - HALO_RADIUS);
        LOAD(local_x + blockDim.x, local_y - HALO_RADIUS, global_x + blockDim.x,
             global_y - HALO_RADIUS);
        LOAD(local_x + blockDim.x, local_y + blockDim.y, global_x + blockDim.x,
             global_y + blockDim.y);
        LOAD(local_x - HALO_RADIUS, local_y + blockDim.y,
             global_x - HALO_RADIUS, global_y + blockDim.y);
      }

    }; // end in_bounds

    __syncthreads();

    /*
    if (in_bounds) {
      FireSimpleCuda f = shared[local_x + local_y * shared_width].fire;
      assert(!is_fire_null(f));
    };

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      bool good = true;
      for (int i = 0; i < blockDim.x + HALO_RADIUS * 2; i++)
        for (int j = 0; j < blockDim.y + HALO_RADIUS * 2; j++) {
          if (!((i + blockDim.x * blockIdx.x) < width &&
                (j + blockDim.x * blockIdx.x) < height))
            continue;
          bool expect_top_halo = blockIdx.y == 0;
          bool expect_bottom_halo = blockIdx.y == gridDim.y - 1;
          bool expect_left_halo = blockIdx.x == 0;
          bool expect_right_halo = blockIdx.x == gridDim.y - 1;
          FireSimpleCuda f = shared[i + j * shared_width].fire;
          if (i < HALO_RADIUS && expect_left_halo != is_fire_null(f)) {
            printf("caca left %d %d\n%d %d\n%f %f %f\n", i, j, blockIdx.x,
                   blockIdx.y, f.azimuth_max, f.speed_max, f.eccentricity);
            printf("fire null? %d\n", is_fire_null(f));
            good = false;
          }
          if (i >= blockDim.x + HALO_RADIUS &&
              expect_right_halo != is_fire_null(f)) {
            printf("caca right %d %d\n%d %d\n%f %f %f\n", i, j, blockIdx.x,
                   blockIdx.y, f.azimuth_max, f.speed_max, f.eccentricity);
            printf("fire null? %d\n", is_fire_null(f));
            good = false;
          }
          if (j < HALO_RADIUS && expect_top_halo != is_fire_null(f)) {
            printf("caca top %d %d\n%d %d\n%f %f %f\n", i, j, blockIdx.x,
                   blockIdx.y, f.azimuth_max, f.speed_max, f.eccentricity);
            printf("fire null? %d\n", is_fire_null(f));
            good = false;
          }
          if (j >= blockDim.y + HALO_RADIUS &&
              expect_bottom_halo != is_fire_null(f)) {
            printf("caca bottom %d %d\n%d %d\n%f %f %f\n", i, j, blockIdx.x,
                   blockIdx.y, f.azimuth_max, f.speed_max, f.eccentricity);
            printf("fire null? %d\n", is_fire_null(f));
            good = false;
          }
        };
      assert(good);
    };
    return;
    */

    ////////////////////////////////////////////////////////////////////////
    // Begin neighbor analysys
    ////////////////////////////////////////////////////////////////////////
    Point me = shared[local_ix];
    FireSimpleCuda fire = me.fire;
    bool is_new = !(me.time < MAX_TIME);

    if (in_bounds && is_new) {
      assert(!is_fire_null(fire));
    };

    Point best = Point_NULL;

    if (in_bounds && is_new) {
#pragma unroll
      for (int i = -1; i < 2; i++) {
#pragma unroll
        for (int j = -1; j < 2; j++) {
          // Skip self and out-of-bounds neighbors
          if (i == 0 && j == 0)
            continue;
          int2 neighbor_pos = make_int2((int)idx_2d.x + i, (int)idx_2d.y + j);
          if (!(neighbor_pos.x >= 0 && neighbor_pos.y >= 0 &&
                neighbor_pos.x < width && neighbor_pos.y < height))
            continue;

          // Good neighbor
          Point neighbor_point =
              shared[(local_x + i) + (local_y + j) * shared_width];

          if (!(neighbor_point.time < MAX_TIME)) {
            // not burning, skip it
            continue;
          };

          // Check if neighbor's reference is usable
          PointRef reference = neighbor_point.reference;
          if (!(reference.time < MAX_TIME && reference.pos.x != SIZE_MAX &&
                reference.pos.y != SIZE_MAX)) {
            printf("referencia ta %d %d %f %d %d %f %f %f\n", reference.pos.x,
                   reference.pos.y, reference.time, idx_2d.x, idx_2d.y,
                   reference.fire.speed_max, reference.fire.azimuth_max,
                   reference.fire.eccentricity);
            assert(false);
          }

          assert(!(reference.pos.x == idx_2d.x && reference.pos.y == idx_2d.y));

          int2 dir = neighbor_direction(
              idx_2d, make_uint2(reference.pos.x, reference.pos.y));
          if (idx_2d.x + dir.x >= 0 && idx_2d.x + dir.x < width &&
              idx_2d.y + dir.y >= 0 && idx_2d.y + dir.y < height) {
            Point possible_blockage =
                shared[(local_x + dir.x) + (local_y + dir.y) * shared_width];
            if (is_point_null(possible_blockage)) {
              // If we haven't analyzed the blockage point yet then we can't use
              // the reference in this iteration
              reference = PointRef_NULL;
            } else {
              assert(!is_fire_null(possible_blockage.fire));
              if (!similar_fires_(possible_blockage.fire, reference.fire)) {
                reference = PointRef_NULL;
              };
            };
          };

          // Look for a new candidate for best time
          Point candidate = Point_NULL;
          if (!is_fire_null(fire) && reference.time < MAX_TIME) {
            // We are combustible
            if (!similar_fires_(fire, reference.fire)) {
              // we can't reuse reference because fire is different to ours.
              // Try to use neighbor as reference
              printf("%d %d %d %d %f %f %f %f %f %f %f\n", idx_2d.x, idx_2d.y,
                     reference.pos.x, reference.pos.y, fire.speed_max,
                     fire.azimuth_max, fire.eccentricity,
                     reference.fire.speed_max, reference.fire.azimuth_max,
                     reference.fire.eccentricity, reference.time);
              PointRef r = PointRef(neighbor_point.time, neighbor_pos,
                                    neighbor_point.fire);
              if (!(r.pos.x == idx_2d.x && r.pos.y == idx_2d.y) &&
                  similar_fires_(fire, r.fire)) {
                reference = r;
              };
            };
            if (reference.time < MAX_TIME) {
              float t = time_to(settings.geo_ref, reference, idx_2d);
              if (t < settings.max_time) {
                candidate.time = t;
                candidate.fire = fire;
                candidate.reference = reference;
              }
            };
          } else if (reference.time < MAX_TIME) {
            // We are not combustible but reference can be used.
            // We assign an access time but a None fire
            float t = time_to(settings.geo_ref, reference, idx_2d);
            if (t < settings.max_time) {
              candidate.time = t;
              candidate.fire = FireSimpleCuda_NULL;
              candidate.reference = reference;
            }
          }

          // If no candidate or candidate improves use it as best
          if (candidate.time < best.time) {
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
      // printf("best time %f\n", best.time);
      assert(global_ix < settings.geo_ref.width * settings.geo_ref.height);
      time[global_ix] = best.time;
      refs_x[global_ix] = best.reference.pos.x;
      refs_y[global_ix] = best.reference.pos.y;
      improved = true;
    };

    ///////////////////////////////////////////////////
    // Wait for other threads to end their analysis and
    // check if any has improved. Then if we're the first
    // thread of the block mark progress
    ///////////////////////////////////////////////////
    any_improved = __syncthreads_count(improved);
    if (improved) {
      shared[local_x + local_y * shared_width] = best;
    }
    __syncthreads();

    if (any_improved && threadIdx.x == 0 && threadIdx.y == 0) {
      size_t block_ix = blockIdx.x + blockIdx.y * gridDim.x;
      progress[block_ix] = 1;
    };
  } while (any_improved);
}

#ifdef __cplusplus
}
#endif
