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

  volatile float *time_;
  volatile unsigned short *refs_x_;
  volatile unsigned short *refs_y_;

  __shared__ Point *shared_;

public:
  __device__ Propagator(const Settings &settings, const unsigned grid_x,
                        const unsigned grid_y, const float *speed_max,
                        const float *azimuth_max, const float *eccentricity,
                        float volatile *time, unsigned short volatile *refs_x,
                        unsigned short volatile *refs_y,
                        __shared__ Point *shared)
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
        refs_x_(refs_x), refs_y_(refs_y), shared_(shared) {
    ASSERT(block_ix_ < gridDim.x * gridDim.y);
  };

  __device__ void run(unsigned *worked, volatile unsigned *progress) const {
    __shared__ bool grid_improved;
    bool first_iteration = true;
    do { // Grid loop
      bool repeat_block = false;
      // Initialize progress to no-progress
      mark_progress(progress, 0);
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        grid_improved = false;
      };
      do { // Block loop
        // First phase, load data from global to shared memory
        load_points_into_shared_memory(true);
        first_iteration = false;

        __syncthreads();

        // Begin neighbor analysys
        Point best;
        find_neighbor_with_least_access_time(&best);

        // End of neighbor analysys, save point if improves
        bool improved = update_point(best);

        // Wait for other threads to end their analysis and
        // check if any has improved. Then if we're the first
        // thread of the block mark progress
        repeat_block = __syncthreads_or(improved);
        if (improved) {
          update_shared_point(best);
        }
        if (repeat_block) {
          mark_progress(progress, 1);
        };
      } while (repeat_block); // end block loop

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

private:
  __device__ inline void mark_progress(volatile unsigned *progress,
                                       unsigned v) const {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      progress[block_ix_] = v;
    }
  }
  __device__ inline void mark_progress_or(volatile unsigned *progress,
                                          unsigned v) const {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      progress[block_ix_] |= v;
    }
  }

  __device__ inline bool update_point(Point p) const {
    Point me = shared_[local_ix_];
    if (in_bounds_ && p.time < MAX_TIME &&
        (is_point_null(me) || p.time < me.time)) {
      // printf("best time %f\n", best.time);
      time_[global_ix_] = p.time;
      refs_x_[global_ix_] = p.reference.pos.x;
      refs_y_[global_ix_] = p.reference.pos.y;
      return true;
    }
    return false;
  }

  __device__ inline void update_shared_point(Point p) const {
    shared_[local_ix_] = p;
  }

  __device__ inline void
  load_points_into_shared_memory(bool first_iteration) const {
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

  __device__ inline void load_point_at_offset(const int2 offset) const {
    int2 local = make_int2(local_x_ + offset.x, local_y_ + offset.y);
    int2 global = make_int2(global_x_ + offset.x, global_y_ + offset.y);
    if (global.x >= 0 && global.x < settings_.geo_ref.width && global.y >= 0 &&
        global.y < settings_.geo_ref.height) {
      uint2 pos = make_uint2(global.x, global.y);
      size_t idx = global.x + global.y * settings_.geo_ref.width;
      Point p = Point(time_[idx],
                      load_fire(idx, speed_max_, azimuth_max_, eccentricity_),
                      PointRef());

      if (p.time < MAX_TIME) {
        __threadfence();
        p.reference.pos = make_ushort2(refs_x_[idx], refs_y_[idx]);
        ASSERT(!(p.reference.pos.x == USHRT_MAX ||
                 p.reference.pos.y == USHRT_MAX));
        ASSERT(p.reference.pos.x < settings_.geo_ref.width ||
               p.reference.pos.y < settings_.geo_ref.height);
        size_t ref_ix =
            p.reference.pos.x + p.reference.pos.y * settings_.geo_ref.width;

        if (p.reference.pos.x == pos.x && p.reference.pos.y == pos.y) {
          p.reference.fire = p.fire;
          p.reference.time = p.time;
        } else {
          p.reference.time = time_[ref_ix];
          p.reference.fire =
              load_fire(ref_ix, speed_max_, azimuth_max_, eccentricity_);
        };
        ASSERT(p.reference.time != MAX_TIME);
      };
      shared_[local.x + local.y * shared_width_] = p;
    } else {
      shared_[local.x + local.y * shared_width_] = Point_NULL;
    }
  }

  __device__ inline void
  find_neighbor_with_least_access_time(Point *result) const {
    Point me = shared_[local_ix_];
    FireSimpleCuda fire = me.fire;
    bool is_new = !(me.time < MAX_TIME);

    Point best = Point_NULL;

    if (is_new && in_bounds_) {
#pragma unroll
      for (int j = -1; j < 2; j++) {
#pragma unroll
        for (int i = -1; i < 2; i++) {
          // Skip self
          if (i == 0 && j == 0)
            continue;

          int2 neighbor_pos = make_int2((int)idx_2d_.x + i, (int)idx_2d_.y + j);

          // skip out-of-bounds neighbors
          if (!(neighbor_pos.x >= 0 && neighbor_pos.y >= 0 &&
                neighbor_pos.x < settings_.geo_ref.width &&
                neighbor_pos.y < settings_.geo_ref.height))
            continue;

          Point neighbor =
              shared_[(local_x_ + i) + (local_y_ + j) * shared_width_];

          if (!(neighbor.time < MAX_TIME && !is_fire_null(neighbor.fire))) {
            // not burning, skip it
            continue;
          };

          PointRef reference = neighbor.reference;
          if ((reference.pos.x == idx_2d_.x && reference.pos.y == idx_2d_.y)) {
            continue;
          }
          ASSERT((reference.time < MAX_TIME && reference.pos.x != USHRT_MAX &&
                  reference.pos.y != USHRT_MAX));
          ASSERT(!is_fire_null(reference.fire));

          // Check if neighbor's reference is usable
          int2 dir = neighbor_direction(
              idx_2d_, make_uint2(reference.pos.x, reference.pos.y));
          ASSERT(((int)idx_2d_.x + dir.x) >= 0 &&
                 ((int)idx_2d_.x + dir.x) < settings_.geo_ref.width &&
                 ((int)idx_2d_.y + dir.y) >= 0 &&
                 ((int)idx_2d_.y + dir.y) < settings_.geo_ref.height);

          Point possible_blockage =
              shared_[(local_x_ + dir.x) + (local_y_ + dir.y) * shared_width_];

          if (is_point_null(possible_blockage)) {
            // If we haven't analyzed the blockage point yet then we can't
            // use the reference in this iteration
            reference = PointRef_NULL;
          } else {
            if (!similar_fires_(possible_blockage.fire, reference.fire)) {
              reference = PointRef(neighbor.time, neighbor_pos, neighbor.fire);
            };
          };

          Point candidate = Point_NULL;
          if (reference.time < MAX_TIME) {
            float t = time_to(settings_.geo_ref, reference, idx_2d_);
            if (t < settings_.max_time && (is_new || t < me.time)) {
              ASSERT(!(reference.pos.x == USHRT_MAX ||
                       reference.pos.y == USHRT_MAX));
              candidate = Point(t, fire, reference);
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

  Propagator sim = Propagator(settings, grid_x, grid_y, speed_max, azimuth_max,
                              eccentricity, time, refs_x, refs_y, shared);

  sim.run(worked, progress);
}

#ifdef __cplusplus
}
#endif
