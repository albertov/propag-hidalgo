use core::ops::Div;
use cuda_std::prelude::*;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::Angle;
use firelib_rs::*;
use geometry::{Coord, GeoReference};
use uom::si::angle::{degree, radian};

const BLOCK_WIDTH: usize = 32;
const BLOCK_HEIGHT: usize = 32;
const SHARED_SIZE: usize = BLOCK_WIDTH * BLOCK_HEIGHT;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn propag(
    geo_ref: GeoReference<f64>,
    max_time: float::T,
    model: &[usize],
    d1hr: &[float::T],
    d10hr: &[float::T],
    d100hr: &[float::T],
    herb: &[float::T],
    wood: &[float::T],
    wind_speed: &[float::T],
    wind_azimuth: &[float::T],
    slope: &[float::T],
    aspect: &[float::T],
    speed_max: *mut float::T,
    azimuth_max: *mut float::T,
    eccentricity: *mut float::T,
    time: *mut float::T,
    refs_x: *mut u32,
    refs_y: *mut u32,
) {
    // Arrays in shared memory for fast analysis within block
    let shared_time = shared_array![float::T; SHARED_SIZE];
    let shared_fire = shared_array![FireSimple; SHARED_SIZE];
    let shared_ref_fire = shared_array![FireSimple; SHARED_SIZE];
    let shared_ref = shared_array![Coord<u32>; SHARED_SIZE];

    // Dimensions of the total area
    let width = geo_ref.size[0] as u32;
    let height = geo_ref.size[1] as u32;

    // Index of this thread into total area
    let idx_2d = thread::index_2d();

    // If this thread is outside the work area, exit early
    if idx_2d.x >= width || idx_2d.y >= height {
        return;
    }

    // Our position in global pixel coords
    let pos = Coord {
        x: idx_2d.x,
        y: idx_2d.y,
    };

    // Our linear index
    let idx = (idx_2d.x + idx_2d.y * width) as usize;

    // Our index into the shared area
    let shared_ix = thread::thread_idx_x() as usize + thread::thread_idx_y() as usize * BLOCK_WIDTH;

    // FIXME: Loop until no thread has worked
    for _ in (0..1000) {
        ////////////////////////////////////////////////////////////////////////
        // First phase, load data from global to shared memory
        ////////////////////////////////////////////////////////////////////////

        // Read our ref's position from global memory and save it into shared
        let sref = Coord {
            x: *refs_x.wrapping_add(idx),
            y: *refs_y.wrapping_add(idx),
        };
        *shared_ref.wrapping_add(shared_ix) = sref;

        // Read our ref's fire from global memory and save it into shared
        let idx_ref = (sref.x + sref.y * width) as usize;
        *shared_ref_fire.wrapping_add(shared_ix) = (FireSimpleCuda {
            azimuth_max: *azimuth_max.wrapping_add(idx_ref),
            speed_max: *speed_max.wrapping_add(idx_ref),
            eccentricity: *eccentricity.wrapping_add(idx_ref),
        })
        .into();

        // Read our time from global memory and save it into shared
        *shared_time.wrapping_add(shared_ix) = *time.wrapping_add(idx);
        // Read our fire from global memory and save it into shared
        *shared_fire.wrapping_add(shared_ix) = (FireSimpleCuda {
            azimuth_max: *azimuth_max.wrapping_add(idx),
            speed_max: *speed_max.wrapping_add(idx),
            eccentricity: *eccentricity.wrapping_add(idx),
        })
        .into();

        // Wait until all other threads have reached this point so we can safely
        // read from shared memory
        thread::sync_threads();

        ////////////////////////////////////////////////////////////////////////
        // Begin neighbor analysys
        ////////////////////////////////////////////////////////////////////////
        for d_i in -1..2 {
            for d_j in -1..2 {
                if d_i == 0 && d_j == 0 {
                    // Don't analyze ourselves
                    continue;
                }
                let n_i = thread::thread_idx_x() as i32 + d_i;
                let n_j = thread::thread_idx_y() as i32 + d_j;
                // Neighbor global pos
                let n_pos = Coord {
                    x: pos.x as i32 + d_i,
                    y: pos.y as i32 + d_j,
                };
                // If this neighbor is outside this block or global area
                // continue
                if n_i < 0
                    || n_i >= BLOCK_WIDTH as _
                    || n_j < 0
                    || n_j >= BLOCK_HEIGHT as _
                    || n_pos.x < 0
                    || n_pos.x >= width as _
                    || n_pos.y < 0
                    || n_pos.y >= height as _
                {
                    continue;
                }

                let n_i = n_i as usize;
                let n_j = n_j as usize;
                let n_pos = Coord {
                    x: n_pos.x as usize,
                    y: n_pos.y as usize,
                };
                // Index of our neighbor in shared memory
                let shared_neigh_ix = n_i + n_j * BLOCK_WIDTH;
                // Read neighbor time from shared mem
                let neigh_time: float::T = *shared_time.wrapping_add(shared_neigh_ix);

                if neigh_time < max_time {
                    // neighbor has fire
                    let neigh_ref_pos = *shared_ref.wrapping_add(shared_neigh_ix);
                    let neigh_ref_fire = &(*shared_ref_fire.wrapping_add(shared_neigh_ix));
                    let bearing = Angle::new::<radian>(geo_ref.bearing(
                        Coord {
                            x: neigh_ref_pos.x as _,
                            y: neigh_ref_pos.y as _,
                        },
                        Coord {
                            x: pos.x as _,
                            y: pos.y as _,
                        },
                    ) as _);
                    //TODO
                }
            }
        }

        let terrain = TerrainCuda {
            d1hr: d1hr[idx],
            d10hr: d10hr[idx],
            d100hr: d100hr[idx],
            herb: herb[idx],
            wood: wood[idx],
            wind_speed: wind_speed[idx],
            wind_azimuth: wind_azimuth[idx],
            slope: slope[idx],
            aspect: aspect[idx],
        };
        if let Some(fire) = Catalog::STANDARD.burn_simple(model[idx], &terrain.into()) {
            let fire = Into::<FireSimpleCuda>::into(fire);
            *time.wrapping_add(idx) = fire.speed_max;
        }
        *refs_x.wrapping_add(idx) = pos.x;
        *refs_y.wrapping_add(idx) = pos.y;
    }
}
