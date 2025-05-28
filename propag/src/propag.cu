#include "propag.h"

//////////////////////////////////////////////////////////////////////////////
/// Propagator
//////////////////////////////////////////////////////////////////////////////
class ALIGN Propagator {
  const Settings &settings_;
  const uint2 gridIx_;
  const uint2 idx_2d_;
  const size_t global_ix_;
  const bool in_bounds_;
  const size_t block_ix_;
  const int global_x_;
  const int global_y_;
  const int local_x_;
  const int local_y_;
  const int shared_width_;
  const int local_ix_;

  const float *speed_max_;
  const float *azimuth_max_;
  const float *eccentricity_;

  float volatile *time_;
  unsigned short volatile *refs_x_;
  unsigned short volatile *refs_y_;
  unsigned volatile *progress_;
  unsigned *worked_;

  __shared__ Point *shared_;
  bool volatile *grid_improved_;

public:
  __device__
  Propagator(const Settings &settings, const unsigned grid_x,
             const unsigned grid_y, unsigned *worked, const float *speed_max,
             const float *azimuth_max, const float *eccentricity,
             float volatile *time, unsigned short volatile *refs_x,
             unsigned short volatile *refs_y, unsigned volatile *progress,
             __shared__ Point *shared, bool volatile *grid_improved)
      : settings_(settings), gridIx_(make_uint2(grid_x, grid_y)),
        idx_2d_(index_2d(gridIx_)),
        global_ix_(idx_2d_.x + idx_2d_.y * settings_.geo_ref.width),
        in_bounds_(idx_2d_.x < settings_.geo_ref.width &&
                   idx_2d_.y < settings_.geo_ref.height),
        block_ix_(blockIdx.x + blockIdx.y * gridDim.x), global_x_(idx_2d_.x),
        global_y_(idx_2d_.y), local_x_(threadIdx.x + HALO_RADIUS),
        local_y_(threadIdx.y + HALO_RADIUS),
        shared_width_(blockDim.x + HALO_RADIUS * 2),
        local_ix_(local_x_ + local_y_ * shared_width_), speed_max_(speed_max),
        azimuth_max_(azimuth_max), eccentricity_(eccentricity), time_(time),
        refs_x_(refs_x), refs_y_(refs_y), progress_(progress), shared_(shared),
        grid_improved_(grid_improved) {
    ASSERT(block_ix_ < gridDim.x * gridDim.y);
  };

  __device__ void run() {};

  // private:
  __device__ inline void load_points_into_shared_memory(bool first_iteration) {
    if (in_bounds_) {
      // Load the central block data
      if (first_iteration) {
        load_point_at_offset(make_int2(0, 0));
      }

      // Load the halo regions
      bool x_near_x0 = threadIdx.x < HALO_RADIUS;
      bool y_near_y0 = threadIdx.y < HALO_RADIUS;
      if (y_near_y0) {
        // Top halo
        load_point_at_offset(make_int2(0, -HALO_RADIUS));
        // Bottom halo
        load_point_at_offset(make_int2(0, blockDim.y));
      }
      if (x_near_x0) {
        // Left halo
        load_point_at_offset(make_int2(-HALO_RADIUS, 0));
        // Right halo
        load_point_at_offset(make_int2(blockDim.x, 0));
      }
      if (x_near_x0 && y_near_y0) {
        // corners
        load_point_at_offset(make_int2(-HALO_RADIUS, -HALO_RADIUS));
        load_point_at_offset(make_int2(blockDim.x, -HALO_RADIUS));
        load_point_at_offset(make_int2(blockDim.x, blockDim.y));
        load_point_at_offset(make_int2(-HALO_RADIUS, blockDim.y));
      }
    }
  }

  __device__ inline void load_point_at_offset(const int2 offset) {
    int2 local = make_int2(local_x_ + offset.x, local_y_ + offset.y);
    int2 global = make_int2(global_x_ + offset.x, global_y_ + offset.y);
    if (global.x >= 0 && global.x < settings_.geo_ref.width && global.y >= 0 &&
        global.y < settings_.geo_ref.height) {
      shared_[local.x + local.y * shared_width_] =
          load_point(settings_.geo_ref, make_uint2(global.x, global.y),
                     global.x + global.y * settings_.geo_ref.width, speed_max_,
                     azimuth_max_, eccentricity_, time_, refs_x_, refs_y_);
    } else {
      shared_[local.x + local.y * shared_width_] = Point_NULL;
    }
  }

  __device__ inline void find_neighbor_with_least_access_time(Point *result) {
    ////////////////////////////////////////////////////////////////////////
    // Begin neighbor analysys
    ////////////////////////////////////////////////////////////////////////
    Point me = shared_[local_ix_];
    FireSimpleCuda fire = me.fire;
    bool is_new = !(me.time < MAX_TIME);

    Point best = Point_NULL;

    if (in_bounds_ && is_new) {
#pragma unroll
      for (int j = -1; j < 2; j++) {
#pragma unroll
        for (int i = -1; i < 2; i++) {
          // Skip self and out-of-bounds neighbors
          if (i == 0 && j == 0)
            continue;

          int2 neighbor_pos = make_int2((int)idx_2d_.x + i, (int)idx_2d_.y + j);
          if (!(neighbor_pos.x >= 0 && neighbor_pos.y >= 0 &&
                neighbor_pos.x < settings_.geo_ref.width &&
                neighbor_pos.y < settings_.geo_ref.height))
            continue;

          // Good neighbor
          Point neighbor_point =
              shared_[(local_x_ + i) + (local_y_ + j) * shared_width_];

          if (!(neighbor_point.time < MAX_TIME &&
                !is_fire_null(neighbor_point.fire))) {
            // not burning, skip it
            continue;
          };

          PointRef reference = neighbor_point.reference;
          ASSERT((reference.time < MAX_TIME && reference.pos.x != USHRT_MAX &&
                  reference.pos.y != USHRT_MAX));
          ASSERT(
              !(reference.pos.x == idx_2d_.x && reference.pos.y == idx_2d_.y));
          ASSERT(!is_fire_null(reference.fire));

          // Check if neighbor's reference is usable
          /*
          int2 dir = neighbor_direction(
              idx_2d, make_uint2(reference.pos.x, reference.pos.y));
          if ((int)idx_2d.x + dir.x >= 0 && (int)idx_2d.x + dir.x < width &&
              (int)idx_2d.y + dir.y >= 0 && (int)idx_2d.y + dir.y < height) {
            Point possible_blockage =
                shared[(local_x + dir.x) + (local_y + dir.y) * shared_width];
            if (is_point_null(possible_blockage)) {
              // If we haven't analyzed the blockage point yet then we can't
              // use the reference in this iteration
              // printf("cant analyze %d %d\n", global_x+dir.x, global_y +
              // dir.y);
              reference = PointRef_NULL;
            } else {
              if (!similar_fires_(possible_blockage.fire, reference.fire)) {
                reference = PointRef_NULL;
              };
            };
          };
          */

          // Look for a new candidate for best time
          Point candidate = Point_NULL;
          if (!is_fire_null(fire) && reference.time < MAX_TIME) {
            // We are combustible
            if (!similar_fires_(fire, reference.fire)) {
              assert(false); // FIXME
              // we can't reuse reference because fire is different to ours.
              // Try to use neighbor as reference
              PointRef r = PointRef(neighbor_point.time, neighbor_pos,
                                    neighbor_point.fire);
              if (!is_fire_null(r.fire) && similar_fires_(fire, r.fire)) {
                reference = r;
              } else {
                reference = PointRef_NULL;
              };
            };
            if (reference.time < MAX_TIME && !is_fire_null(reference.fire)) {
              float t = time_to(settings_.geo_ref, reference, idx_2d_);
              if (t < settings_.max_time) {
                ASSERT(!(reference.pos.x == USHRT_MAX ||
                         reference.pos.y == USHRT_MAX));
                candidate.time = t;
                candidate.fire = fire;
                candidate.reference = reference;
              }
            };
          } else if (reference.time < MAX_TIME) {
            assert(false); // FIXME
            // We are not combustible but reference can be used.
            // We assign an access time but a None fire
            float t = time_to(settings_.geo_ref, reference, idx_2d_);
            if (t < settings_.max_time) {
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
    *result = best;
  }
};

#ifdef __cplusplus
extern "C" {
#endif

__global__ void propag(const Settings &settings, unsigned grid_x,
                       unsigned grid_y, unsigned *worked,
                       const float *speed_max, const float *azimuth_max,
                       const float *eccentricity, float volatile *time,
                       unsigned short volatile *refs_x,
                       unsigned short volatile *refs_y,
                       unsigned volatile *progress) {
  extern __shared__ Point shared[];
  __shared__ bool grid_improved;

  Propagator sim = Propagator(settings, grid_x, grid_y, worked, speed_max,
                              azimuth_max, eccentricity, time, refs_x, refs_y,
                              progress, shared, &grid_improved);

  unsigned width = settings.geo_ref.width;
  unsigned height = settings.geo_ref.height;

  uint2 gridIx = make_uint2(grid_x, grid_y);

  uint2 idx_2d = index_2d(gridIx);

  size_t global_ix = idx_2d.x + idx_2d.y * width;

  bool in_bounds = idx_2d.x < width && idx_2d.y < height;

  size_t block_ix = blockIdx.x + blockIdx.y * gridDim.x;
  ASSERT(block_ix < gridDim.x * gridDim.y);

  // Calculate local (shared memory) indices
  int local_x = threadIdx.x + HALO_RADIUS;
  int local_y = threadIdx.y + HALO_RADIUS;
  int shared_width = blockDim.x + HALO_RADIUS * 2;
  int local_ix = local_x + local_y * shared_width;

  bool first_iteration = true;
  do { // Grid loop
    bool any_improved = false;
    // Initialize progress to no-progress
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      progress[block_ix] = 0;
      grid_improved = false;
    };
    do { // Block loop
      //////////////////////////////////////////////////////
      // First phase, load data from global to shared memory
      //////////////////////////////////////////////////////
      sim.load_points_into_shared_memory(first_iteration);
      first_iteration = false;

      __syncthreads();

      ////////////////////////////////////////////////////////////////////////
      // Begin neighbor analysys
      ////////////////////////////////////////////////////////////////////////
      Point best;
      sim.find_neighbor_with_least_access_time(&best);

      ///////////////////////////////////////////////////
      // End of neighbor analysys, save point if improves
      ///////////////////////////////////////////////////
      bool improved = in_bounds && best.time < MAX_TIME;
      if (improved) {
        // printf("best time %f\n", best.time);
        ASSERT(global_ix < settings.geo_ref.width * settings.geo_ref.height);
        time[global_ix] = best.time;
        refs_x[global_ix] = best.reference.pos.x;
        refs_y[global_ix] = best.reference.pos.y;
      };

      ///////////////////////////////////////////////////
      // Wait for other threads to end their analysis and
      // check if any has improved. Then if we're the first
      // thread of the block mark progress
      ///////////////////////////////////////////////////
      any_improved = __syncthreads_or(improved);
      if (improved) {
        // write our Point to shared mem *after* __syncthreads_or
        shared[local_x + local_y * shared_width] = best;
      }
      if (any_improved && threadIdx.x == 0 && threadIdx.y == 0) {
        progress[block_ix] = 1;
      };
      __syncthreads();
    } while (any_improved); // end block loop

    // Block has finished. Check if others have too and set grid_improved.
    // Analysys ends when grid has not improved
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      grid_improved = false;
      for (int i = 0; i < gridDim.x * gridDim.y; i++) {
        grid_improved |= progress[i];
      }
      if (grid_improved && blockIdx.x == 0 && blockIdx.y == 0) {
        *worked = 1;
      }
    }
    grid.sync();
  } while (grid_improved); // end grid loop
}

#ifdef __cplusplus
}
#endif
