use core::ops::Div;
use cuda_std::prelude::*;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::Angle;
use firelib_rs::*;
use geometry::{Coord, CoordFloat, GeoReference};
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
    speed_max: *mut Option<float::T>,
    azimuth_max: *mut Option<float::T>,
    eccentricity: *mut Option<float::T>,
    time: *mut Option<float::T>,
    refs_x: *mut Option<usize>,
    refs_y: *mut Option<usize>,
) {
    // Arrays in shared memory for fast analysis within block
    let shared = shared_array![Option<Point<float::T>>; SHARED_SIZE];

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
        x: idx_2d.x as usize,
        y: idx_2d.y as usize,
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
        let me = Point::<_>::load(
            idx,
            time,
            speed_max,
            azimuth_max,
            eccentricity,
            refs_x,
            refs_y,
        );
        *shared.wrapping_add(shared_ix) = me;

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
                if let Some(neigh) = *shared.wrapping_add(shared_neigh_ix) {
                    // neighbor has fire
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
                        *speed_max.wrapping_add(idx) = Some(fire.speed_max);
                        *azimuth_max.wrapping_add(idx) = Some(fire.azimuth_max);
                        *eccentricity.wrapping_add(idx) = Some(fire.eccentricity);
                    }
                    let neigh_ref = neigh.effective_ref(pos);
                    let bearing = Angle::new::<radian>(geo_ref.bearing(
                        Coord {
                            x: neigh_ref.pos.x as _,
                            y: neigh_ref.pos.y as _,
                        },
                        Coord {
                            x: pos.x as _,
                            y: pos.y as _,
                        },
                    ) as _);
                    let can_use_ref = {
                        let pos = Coord {
                            x: pos.x as float::T,
                            y: pos.y as float::T,
                        };
                        let other = Coord {
                            x: neigh_ref.pos.x as float::T,
                            y: neigh_ref.pos.y as float::T,
                        };
                        if let Some(possible_blockage) = geometry::line_to(pos, other).nth(1) {
                            let shared_blockage_ix = {
                                if possible_blockage.x >= 0.0
                                    && possible_blockage.x < BLOCK_WIDTH as float::T
                                    && possible_blockage.y >= 0.0
                                    && possible_blockage.y < BLOCK_HEIGHT as float::T
                                {
                                    Some(
                                        possible_blockage.x as usize
                                            + possible_blockage.y as usize * BLOCK_WIDTH,
                                    )
                                } else {
                                    None
                                }
                            };
                            //TODO
                            false
                        } else {
                            false
                        }
                    };
                }
            }
        }

        *refs_x.wrapping_add(idx) = Some(pos.x);
        *refs_y.wrapping_add(idx) = Some(pos.y);
    }
}

#[derive(Copy, Clone)]
struct PointRef<T>
where
    T: CoordFloat,
{
    time: T,
    pos: Coord<usize>,
    fire: FireSimple,
}

#[derive(Copy, Clone)]
struct Point<T>
where
    T: CoordFloat,
{
    time: T,
    fire: FireSimple,
    reference: Option<PointRef<T>>,
}

impl Point<float::T> {
    unsafe fn load(
        idx: usize,
        time: *const Option<float::T>,
        speed_max: *const Option<float::T>,
        azimuth_max: *const Option<float::T>,
        eccentricity: *const Option<float::T>,
        ref_x: *const Option<usize>,
        ref_y: *const Option<usize>,
    ) -> Option<Self> {
        let p_time = (*time.wrapping_add(idx))?;
        let fire = (Some(FireSimpleCuda {
            azimuth_max: (*azimuth_max.wrapping_add(idx))?,
            speed_max: (*speed_max.wrapping_add(idx))?,
            eccentricity: (*eccentricity.wrapping_add(idx))?,
        }))?
        .into();
        let ref_pos: Option<Coord<usize>> = Some(Coord {
            x: (*ref_x.wrapping_add(idx))?,
            y: (*ref_y.wrapping_add(idx))?,
        });
        let reference: Option<PointRef<float::T>> = ref_pos.and_then(|ref_pos| {
            let ref_ix: usize = ref_pos.x + ref_pos.y * BLOCK_WIDTH;
            let time = (*time.wrapping_add(ref_ix))?;
            let fire = Some(FireSimpleCuda {
                azimuth_max: (*azimuth_max.wrapping_add(ref_ix))?,
                speed_max: (*speed_max.wrapping_add(ref_ix))?,
                eccentricity: (*eccentricity.wrapping_add(ref_ix))?,
            })?
            .into();
            Some(PointRef {
                time,
                pos: ref_pos,
                fire,
            })
        });
        Some(Point {
            time: p_time,
            fire,
            reference,
        })
    }

    fn effective_ref(&self, my_pos: Coord<usize>) -> PointRef<float::T> {
        self.reference.unwrap_or(PointRef {
            pos: my_pos,
            time: self.time,
            fire: self.fire,
        })
    }
}
