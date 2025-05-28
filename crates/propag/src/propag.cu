#include "propag.h"

__device__ static inline bool
is_boundary(const unsigned int *__restrict__ boundaries, int ix) {
  int byte_ix = ix >> 5;
  int bit_ix = ix - (byte_ix << 5);
  return boundaries[byte_ix] & (1 << bit_ix);
}
__device__ static inline void set_boundary(unsigned int *boundaries, int ix) {
  int byte_ix = ix >> 5;
  int bit_ix = ix - (byte_ix << 5);
  atomicOr(boundaries + byte_ix, 1 << bit_ix);
}

//////////////////////////////////////////////////////////////////////////////
/// Propagator
//////////////////////////////////////////////////////////////////////////////
class Propagator {
  const Settings &settings_;
  const uint2 gridIx_;
  const uint2 idx_2d_;
  const size_t global_ix_;
  const bool in_bounds_;
  const size_t block_ix_;
  const int local_x_;
  const int local_y_;
  const int shared_width_;
  const int local_ix_;

  const float *const speed_max_;
  const float *const azimuth_max_;
  const float *const eccentricity_;

  volatile float *time_;
  volatile unsigned short *refs_x_;
  volatile unsigned short *refs_y_;

  Point *shared_;

public:
  __device__ Propagator(const Settings settings, const unsigned grid_x,
                        const unsigned grid_y, const float *speed_max,
                        const float *azimuth_max, const float *eccentricity,
                        float volatile *time, unsigned short volatile *refs_x,
                        unsigned short volatile *refs_y, Point *shared)
      : settings_(settings), gridIx_(make_uint2(grid_x, grid_y)),
        idx_2d_(index_2d(gridIx_)),
        global_ix_(idx_2d_.x + idx_2d_.y * settings_.geo_ref.width),
        in_bounds_(idx_2d_.x < settings_.geo_ref.width &&
                   idx_2d_.y < settings_.geo_ref.height),
        block_ix_(blockIdx.x + blockIdx.y * gridDim.x),
        local_x_(threadIdx.x + HALO_RADIUS),
        local_y_(threadIdx.y + HALO_RADIUS),
        shared_width_(blockDim.x + HALO_RADIUS * 2),
        local_ix_(local_x_ + local_y_ * shared_width_), speed_max_(speed_max),
        azimuth_max_(azimuth_max), eccentricity_(eccentricity), time_(time),
        refs_x_(refs_x), refs_y_(refs_y), shared_(shared) {
    ASSERT(block_ix_ < gridDim.x * gridDim.y);
    ASSERT(!(settings_.geo_ref.width < 1 || settings_.geo_ref.height < 1));
  };

  __device__ void run(unsigned *worked, unsigned *progress,
                      const unsigned int *__restrict__ boundaries) {
    __shared__ bool grid_improved;
    do { // Grid loop
      if (threadIdx.x + threadIdx.y == 0) {
        // Reset the block-local grid_improved flag if we're the block leader
        grid_improved = false;
        if (blockIdx.x + blockIdx.y == 0) {
          // Reset the grid-global progress flag if we're also the grid leader
          *progress = 0;
        }
      };
      // First phase, load data from global to shared memory.
      load_points_into_shared_memory();

      // Begin neighbor analysys
      Point best = find_neighbor_which_reaches_first(boundaries);

      cooperative_groups::this_grid().sync();
      // End of neighbor analysys, update point in global and shared memory
      // if it improves
      bool improved = update_point(best);
      // Sync to wait for other threads to end their analysis and
      // check if any has improved. Then if we're the first
      // thread of the block mark progress
      bool block_improved = __syncthreads_or(improved);
      if (block_improved && threadIdx.x + threadIdx.y == 0) {
        // Signal that at least one block in this grid has progressed.
        // No need for atomic here because it's enough to flip it != 0
        *progress = 1;
      }

      // Block has finished. Check if others have too after syncing grid to
      // set shared grid_improved in (shared mem) for the rest of the block.
      cooperative_groups::this_grid().sync();
      if (threadIdx.x + threadIdx.y == 0) {
        // Update grid_improved flag if we're the block leader
        grid_improved = *progress > 0;
        if (grid_improved && blockIdx.x + blockIdx.y == 0) {
          // If we're also the grid leader, signal the kernel launcher
          // that this grid has worked on this round
          *worked = 1;
        }
      }
      // Sync here to wait for the grid_improved flag to have the correct value
      __syncthreads();
      // Analysys ends when grid has not improved
    } while (grid_improved); // end grid loop
  }
  __device__ void post_propagate(const unsigned int *__restrict__ boundaries) {
    if (in_bounds_) {
      Point me = load_point(make_int2(idx_2d_.x, idx_2d_.y));
      if (fire_is_null(me.fire) || me.time > settings_.max_time) {
        time_[global_ix_] = FLT_MAX;
      };
    };
  }
  __device__ void pre_propagate(unsigned int *boundaries) {
    load_points_into_shared_memory();
    find_boundaries(boundaries);
  }

private:
  __device__ inline Point find_neighbor_which_reaches_first(
      const unsigned int *__restrict__ boundaries) const {
    if (in_bounds_) {
      Point best = shared_[local_ix_];
#pragma unroll
      for (int j = -1; j < 2; j++) {
#pragma unroll
        for (int i = -1; i < 2; i++) {
          // Skip self
          if (i == 0 && j == 0)
            continue;

          const int2 neighbor_pos =
              make_int2((int)idx_2d_.x + i, (int)idx_2d_.y + j);

          // skip out-of-bounds neighbors
          if (!pos_in_bounds(neighbor_pos))
            continue;

          const Point neighbor =
              shared_[(local_x_ + i) + (local_y_ + j) * shared_width_];

          // not burning, skip it
          if (!(neighbor.time < settings_.max_time &&
                !fire_is_null(neighbor.fire))) {
            continue;
          };

          PointRef reference = neighbor.reference;
          // The neighbor's reference is me, refuse to burn myself again
          ASSERT(reference.is_valid(settings_.geo_ref));
          if ((reference.pos.x == idx_2d_.x && reference.pos.y == idx_2d_.y)) {
            continue;
          }
          const PointRef neighbor_as_ref(neighbor.time, neighbor_pos);
          const float neighbor_time = time_from(neighbor_as_ref, neighbor.fire);
          float candidate_time = time_from(reference, neighbor.fire);
          if (candidate_time < neighbor_time && candidate_time < best.time) {
            // Check if the path to neighbor's reference is not blocked by a
            // point with a different fire
            int2 possible_blockage_pos;
            DDA iter(neighbor_pos, reference.pos);
            while (iter.next(possible_blockage_pos)) {
              ASSERT(pos_in_bounds(possible_blockage_pos));
              size_t blockage_idx =
                  possible_blockage_pos.x +
                  possible_blockage_pos.y * settings_.geo_ref.width;
              if (is_boundary(boundaries, blockage_idx)) {
                reference = neighbor_as_ref;
                candidate_time = neighbor_time;
                break;
              };
            }
            if (candidate_time < best.time) {
              best.time = candidate_time;
              best.reference = reference;
            }
          } else if (neighbor_time < best.time) {
            best.time = neighbor_time;
            best.reference = neighbor_as_ref;
          };
        };
      };
      return best;
    };
    return Point_NULL;
  }

  __device__ inline bool update_point(Point p) {
    if (p.time < shared_[local_ix_].time) {
      ASSERT(in_bounds_);
      shared_[local_ix_] = p;
      commit_point();
      return true;
    }
    return false;
  }
  __device__ inline void commit_point() {
    const Point &me = shared_[local_ix_];
    ASSERT(me.reference.is_valid(settings_.geo_ref));
    refs_x_[global_ix_] = me.reference.pos.x;
    refs_y_[global_ix_] = me.reference.pos.y;
    time_[global_ix_] = me.time;
  }

  __device__ inline Point load_point(int2 ipos) const {
    if (ipos.x >= 0 && ipos.x < settings_.geo_ref.width && ipos.y >= 0 &&
        ipos.y < settings_.geo_ref.height) {
      uint2 pos = make_uint2(ipos.x, ipos.y);
      size_t idx = ipos.x + ipos.y * settings_.geo_ref.width;
      Point p = Point(time_[idx],
                      load_fire(idx, speed_max_, azimuth_max_, eccentricity_),
                      PointRef());

      if (p.time < FLT_MAX) {
        p.reference.pos = make_ushort2(refs_x_[idx], refs_y_[idx]);
        ASSERT(p.reference.is_valid(settings_.geo_ref));
        int reference_ix =
            p.reference.pos.x + p.reference.pos.y * settings_.geo_ref.width;
        p.reference.time = time_[reference_ix];
        ASSERT(p.reference.time != FLT_MAX);
      };
      return p;
    };
    return Point_NULL;
  };

  __device__ inline void load_points_into_shared_memory() {
    if (in_bounds_) {
      load_point_at_offset(make_int2(0, 0));

      // Load the halo regions
      bool x_near_x0 = threadIdx.x < HALO_RADIUS;
      bool y_near_y0 = threadIdx.y < HALO_RADIUS;
      if (x_near_x0 && y_near_y0) {
        // corners
        load_point_at_offset(make_int2(-HALO_RADIUS, -HALO_RADIUS));
        load_point_at_offset(make_int2(blockDim.x, -HALO_RADIUS));
        load_point_at_offset(make_int2(-HALO_RADIUS, blockDim.y));
        load_point_at_offset(make_int2(blockDim.x, blockDim.y));
      }
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
    }
    // Sync here to wait for the shared Point data to have been
    // updated by other threads
    __syncthreads();
  }

  __device__ inline void load_point_at_offset(const int2 offset) {
    int2 local = make_int2(local_x_ + offset.x, local_y_ + offset.y);
    int2 global = make_int2(idx_2d_.x + offset.x, idx_2d_.y + offset.y);
    shared_[local.x + local.y * shared_width_] = load_point(global);
  }

  __device__ inline float time_from(const PointRef &from,
                                    const FireSimpleCuda &fire) const {
    int2 from_pos = make_int2(from.pos.x, from.pos.y);
    int2 to_pos = make_int2(idx_2d_.x, idx_2d_.y);
    float azimuth;
    if (settings_.geo_ref.transform.gt.dy < 0) {
      // north-up geotransform
      azimuth = atan2f(to_pos.x - from_pos.x, from_pos.y - to_pos.y);
    } else {
      azimuth = atan2f(to_pos.x - from_pos.x, to_pos.y - from_pos.y);
      // south-up geotransform
    };
    // Uses CUDA intrinsics for performance:
    // a * __frcp_rd(x) ~= a / x
    // __fsqrt_rz(x) ~= sqrt(x)
    // __cosf(x) ~= cos(x)
    float angle = abs(azimuth - fire.azimuth_max);
    float denom = (1.0 - fire.eccentricity * __cosf(angle));
    float factor = (1.0 - fire.eccentricity) * __frcp_rd(denom);
    float speed = fire.speed_max * factor;
    ASSERT(speed >= 0.0);
    float dx = (from_pos.x - to_pos.x) * settings_.geo_ref.transform.gt.dx;
    float dy = (from_pos.y - to_pos.y) * settings_.geo_ref.transform.gt.dy;
    float distance = __fsqrt_rz(dx * dx + dy * dy);
    return from.time + (distance * __frcp_rd(speed));
  }

  __device__ inline bool similar_fires(const volatile FireSimpleCuda &a,
                                       const volatile FireSimpleCuda &b) const {
    // TODO: Make configurable
    return (abs(a.speed_max - b.speed_max) < 0.1 &&
            abs(a.azimuth_max - b.azimuth_max) < (5.0 * (2 * M_PI) / 360.0) &&
            abs(a.eccentricity - b.eccentricity) < 0.05);
  }

  __device__ inline bool pos_in_bounds(int2 pos) const {
    return pos.x >= 0 && pos.y >= 0 && pos.x < settings_.geo_ref.width &&
           pos.y < settings_.geo_ref.height;
  }
  __device__ inline bool pos_in_bounds(ushort2 pos) const {
    return pos.x < settings_.geo_ref.width && pos.y < settings_.geo_ref.height;
  }

  __device__ void print_info(const char msg[]) const {
    static const char true_[] = "true";
    static const char false_[] = "false";
    printf("-------------------------------------------------------------\n"
           "%s\n"
           "-------------------------------------------------------------\n"
           "width=%d\n"
           "height=%d\n"
           "gridIx=(%d, %d)\n"
           "blockDim=(%d, %d)\n"
           "idx_2d=(%d, %d)\n"
           "modBlockSize=(%d, %d)\n"
           "local_xy=(%d, %d)\n"
           "global_ix=%  ld\n"
           "in_bounds_=%s\n"
           "block_ix_=%ld\n"
           "shared_width_=%d\n"
           "local_ix_=%d\n"
           "time=%.4f\n"
           "has_fire=%s\n"
           "ref_x=%d\n"
           "ref_y=%d\n",
           msg, settings_.geo_ref.width, settings_.geo_ref.height, gridIx_.x,
           gridIx_.y, blockDim.x, blockDim.y, idx_2d_.x, idx_2d_.y,
           idx_2d_.x % blockDim.x, idx_2d_.y % blockDim.y, local_x_, local_y_,
           global_ix_, (in_bounds_ ? true_ : false_), block_ix_, shared_width_,
           local_ix_, (in_bounds_ ? time_[global_ix_] : FLT_MAX),
           (in_bounds_ ? (speed_max_[global_ix_] != 0.0 ? true_ : false_)
                       : false_),
           (in_bounds_ ? refs_x_[global_ix_] : USHRT_MAX),
           (in_bounds_ ? refs_y_[global_ix_] : USHRT_MAX));
  }

  __device__ inline void find_boundaries(unsigned int *boundaries) {
    if (in_bounds_) {
      const Point me = shared_[local_ix_];
      unsigned short changed = 0;
#pragma unroll
      for (int i = -1; i < 2; i++) {
#pragma unroll
        for (int j = -1; j < 2; j++) {
          // Skip self
          if (i == 0 && j == 0)
            continue;
          int2 neighbor_pos = make_int2((int)idx_2d_.x + i, (int)idx_2d_.y + j);

          // skip out-of-bounds neighbors
          if (!(neighbor_pos.x >= 0 && neighbor_pos.y >= 0 &&
                neighbor_pos.x < settings_.geo_ref.width &&
                neighbor_pos.y < settings_.geo_ref.height))
            continue;

          const Point neighbor =
              shared_[(local_x_ + i) + (local_y_ + j) * shared_width_];

          changed |= !similar_fires(neighbor.fire, me.fire) &&
                     me.fire.speed_max < neighbor.fire.speed_max;
        }
      }
      if (changed) {
        set_boundary(boundaries, global_ix_);
      }
    }
  }
};

#ifdef __cplusplus
extern "C" {
#endif

__global__ void
propag(const Settings settings, const unsigned grid_x, const unsigned grid_y,
       unsigned *worked, const float *const speed_max,
       const float *const azimuth_max, const float *const eccentricity,
       float volatile *time, unsigned short volatile *refs_x,
       unsigned short volatile *refs_y,
       const unsigned int *__restrict__ boundaries, unsigned *progress) {
  extern __shared__ Point shared[];

  Propagator sim(settings, grid_x, grid_y, speed_max, azimuth_max, eccentricity,
                 time, refs_x, refs_y, shared);
  sim.run(worked, progress, boundaries);
}

__global__ void
post_propagate(const Settings settings, const unsigned grid_x,
               const unsigned grid_y, const float *const speed_max,
               const float *const azimuth_max, const float *const eccentricity,
               float volatile *time, unsigned short volatile *refs_x,
               unsigned short volatile *refs_y,
               unsigned int *__restrict__ boundaries) {
  extern __shared__ Point shared[];

  Propagator sim(settings, grid_x, grid_y, speed_max, azimuth_max, eccentricity,
                 time, refs_x, refs_y, shared);
  sim.post_propagate(boundaries);
}

__global__ void
pre_propagate(const Settings settings, const unsigned grid_x,
              const unsigned grid_y, const float *const speed_max,
              const float *const azimuth_max, const float *const eccentricity,
              float volatile *time, unsigned short volatile *refs_x,
              unsigned short volatile *refs_y, unsigned int *boundaries) {
  extern __shared__ Point shared[];

  Propagator sim(settings, grid_x, grid_y, speed_max, azimuth_max, eccentricity,
                 time, refs_x, refs_y, shared);
  sim.pre_propagate(boundaries);
}

#ifdef __cplusplus
}
#endif
