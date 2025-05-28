#include "propag.h"

//////////////////////////////////////////////////////////////////////////////
/// propag
//////////////////////////////////////////////////////////////////////////////

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

  unsigned width = settings.geo_ref.width;
  unsigned height = settings.geo_ref.height;

  uint2 gridIx = make_uint2(grid_x, grid_y);

  uint2 idx_2d = index_2d(gridIx);

  size_t global_ix = idx_2d.x + idx_2d.y * width;

  bool in_bounds = idx_2d.x < width && idx_2d.y < height;

  size_t block_ix = blockIdx.x + blockIdx.y * gridDim.x;
  ASSERT(block_ix < gridDim.x * gridDim.y);

  // Calculate global indices
  int global_x = idx_2d.x;
  int global_y = idx_2d.y;

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
      if (in_bounds) {
        // Load the central block data
        if (first_iteration) {
          LOAD(local_x, local_y, global_x, global_y);
          first_iteration = false;
        }

        // Load the halo regions
        bool x_near_x0 = threadIdx.x < HALO_RADIUS;
        bool y_near_y0 = threadIdx.y < HALO_RADIUS;
        if (y_near_y0) {
          // Top halo
          LOAD(local_x, local_y - HALO_RADIUS, global_x,
               global_y - HALO_RADIUS);
          // Bottom halo
          LOAD(local_x, local_y + blockDim.y, global_x, global_y + (int)blockDim.y);
        }
        if (x_near_x0) {
          // Left halo
          LOAD(local_x - HALO_RADIUS, local_y, global_x - HALO_RADIUS,
               global_y);
          // Right halo
          LOAD(local_x + blockDim.x, local_y, global_x + (int)blockDim.x, global_y);
        }
        if (x_near_x0 && y_near_y0) {
          // corners
          LOAD(local_x - HALO_RADIUS, local_y - HALO_RADIUS,
               global_x - HALO_RADIUS, global_y - HALO_RADIUS);
          LOAD(local_x + blockDim.x, local_y - HALO_RADIUS,
               global_x + (int)blockDim.x, global_y - HALO_RADIUS);
          LOAD(local_x + blockDim.x, local_y + blockDim.y,
               global_x + (int)blockDim.x, global_y + (int)blockDim.y);
          LOAD(local_x - HALO_RADIUS, local_y + blockDim.y,
               global_x - HALO_RADIUS, global_y + (int)blockDim.y);
        }
      }


      __syncthreads();

      ////////////////////////////////////////////////////////////////////////
      // Begin neighbor analysys
      ////////////////////////////////////////////////////////////////////////
      Point me = shared[local_ix];
      FireSimpleCuda fire = me.fire;
      bool is_new = !(me.time < MAX_TIME);

      Point best = Point_NULL;

      if (in_bounds && is_new) {
#pragma unroll
        for (int j = -1; j < 2; j++) {
#pragma unroll
          for (int i = -1; i < 2; i++) {
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

            if (!(neighbor_point.time < MAX_TIME &&
                  !is_fire_null(neighbor_point.fire))) {
              // not burning, skip it
              continue;
            };

            PointRef reference = neighbor_point.reference;
            ASSERT((reference.time < MAX_TIME && reference.pos.x != USHRT_MAX &&
                    reference.pos.y != USHRT_MAX));
            ASSERT(
                !(reference.pos.x == idx_2d.x && reference.pos.y == idx_2d.y));
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
                assert(false); //FIXME
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
                float t = time_to(settings.geo_ref, reference, idx_2d);
                if (t < settings.max_time) {
                  ASSERT(!(reference.pos.x == USHRT_MAX || reference.pos.y == USHRT_MAX));
                  candidate.time = t;
                  candidate.fire = fire;
                  candidate.reference = reference;
                }
              };
            } else if (reference.time < MAX_TIME) {
              assert(false); //FIXME
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
