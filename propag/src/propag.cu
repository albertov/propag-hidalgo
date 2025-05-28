#include <float.h>
#include <stdio.h>
#include "geometry.h"
#include "firelib_cuda.h"

#define MAX_TIME FLT_MAX

__device__ inline uint2 index_2d() {
  return make_uint2(
      threadIdx.x + blockIdx.x * blockDim.x,
      threadIdx.y + blockIdx.y * blockDim.y
      );
}

__device__ inline size_t compute_shared_ix(uint2 pos) {
  size_t shared_x = pos.x % blockDim.x;
  size_t shared_y = pos.y % blockDim.x;
  size_t ix = shared_x
            + shared_y * (blockDim.x + HALO_RADIUS * 2);
  assert(ix < (blockDim.x+HALO_RADIUS*2) * (blockDim.y+HALO_RADIUS*2));
  assert(shared_x < blockDim.x + HALO_RADIUS * 2);
  assert(shared_y < blockDim.y + HALO_RADIUS * 2);
  return ix;
}


__device__ inline bool offset_shared_ix(GeoReference &geo_ref, int2 offset, size_t *result) {
  uint2 upos = index_2d();
  int2 pos = make_int2(upos.x+offset.x, upos.y+offset.y);
  if (pos.x >= 0 && pos.y >= 0 &&
      pos.x < geo_ref.width && pos.y < geo_ref.height) {
    *result = compute_shared_ix(make_uint2(pos.x, pos.y));
    return true;
  } else {
    return false;
  }
}

__device__ inline bool offset_ix(GeoReference &geo_ref, int2 offset, size_t *result) {
  uint2 upos = index_2d();
  int2 pos = make_int2(upos.x+offset.x, upos.y+offset.y);
  if (pos.x >= 0 && pos.y >= 0 &&
      pos.x < geo_ref.width && pos.y < geo_ref.height) {
    *result = pos.x + pos.y * geo_ref.width;
    return true;
  } else {
    return false;
  }
}

__device__ inline FireSimpleCuda load_fire(
    size_t idx,
    const float *speed_max,
    const float *azimuth_max,
    const float *eccentricity
    ) {
  FireSimpleCuda ret = FireSimpleCuda_NULL;
  ret.speed_max = speed_max[idx];
  ret.azimuth_max = azimuth_max[idx];
  ret.eccentricity = eccentricity[idx];
  return ret;
}

__device__ inline Point load_point(
    GeoReference &geo_ref,
    uint2 pos,
    size_t idx,
    const float *speed_max,
    const float *azimuth_max,
    const float *eccentricity,
    const float *time,
    const size_t *ref_x,
    const size_t *ref_y
    ) {
  float p_time = time[idx];
  if (p_time == MAX_TIME) {
    return Point_NULL;
  }
  FireSimpleCuda fire = load_fire(idx, speed_max, azimuth_max, eccentricity);
  uint2 ref_pos = make_uint2(ref_x[idx], ref_y[idx]);
  assert(!(ref_pos.x == SIZE_MAX || ref_pos.y == SIZE_MAX));
  size_t ref_ix = ref_pos.x + ref_pos.y * geo_ref.width;
  FireSimpleCuda ref_fire = load_fire(ref_ix, speed_max, azimuth_max, eccentricity);
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

#ifdef __cplusplus
extern "C" {
#endif

__global__ void propag(
    Settings settings,
    const float *speed_max,
    size_t speed_max_len,
    const float *azimuth_max,
    long azimuth_max_len,
    const float *eccentricity,
    size_t eccentricity_len,
    float volatile *time,
    size_t volatile *refs_x,
    size_t volatile *refs_y,
    unsigned volatile *progress
  )
{
  extern __shared__ Point shared[];

  uint2 idx_2d = index_2d();
  uint2 pos = idx_2d;

  size_t global_ix = idx_2d.x + idx_2d.y * settings.geo_ref.width;
  bool in_bounds = idx_2d.x < settings.geo_ref.width &&
                   idx_2d.y < settings.geo_ref.height;

  size_t block_ix = blockIdx.x +  blockIdx.y * gridDim.x;
  assert(block_ix < gridDim.x * gridDim.y);

  if (threadIdx.x==0 && threadIdx.y==0) {
    progress[block_ix] = 0;
  }


  printf("Hello World from GPU %d %d!\n", settings.geo_ref.width, idx_2d.x);
}

#ifdef __cplusplus
}
#endif
