use core::ops::Div;
use cuda_std::prelude::*;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::Angle;
use firelib_rs::*;
use geometry::{Coord, CoordFloat, GeoReference};
use uom::num_traits::NumCast;
use uom::si::angle::{degree, radian};
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

const BLOCK_WIDTH: usize = 16;
const BLOCK_HEIGHT: usize = 16;
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

    let when_lt_max_time = |p: Point<float::T>| if p.time < max_time { Some(p) } else { None };

    // Dimensions of the total area
    let width = geo_ref.size[0] as u32;
    let height = geo_ref.size[1] as u32;

    // Index of this thread into total area
    let idx_2d = thread::index_2d();

    // are we in-bounds of the target raster?
    let in_bounds = idx_2d.x < width && idx_2d.y < height;

    // Our position in global pixel coords
    let pos = Coord {
        x: idx_2d.x as usize,
        y: idx_2d.y as usize,
    };

    // Our global index
    let global_ix = (idx_2d.x + idx_2d.y * width) as usize;

    // Our index into the shared area
    let shared_ix = thread::thread_idx_x() as usize + thread::thread_idx_y() as usize * BLOCK_WIDTH;

    let mut improved: u32 = 0;

    loop {
        ////////////////////////////////////////////////////////////////////////
        // First phase, load data from global to shared memory
        ////////////////////////////////////////////////////////////////////////
        if in_bounds {
            let me = Point::<float::T>::load(
                global_ix,
                time,
                speed_max,
                azimuth_max,
                eccentricity,
                refs_x,
                refs_y,
            );
            *shared.wrapping_add(shared_ix) = me;
        }
        // Wait until all other threads have reached this point so we can safely
        // read from shared memory
        thread::sync_threads();

        ////////////////////////////////////////////////////////////////////////
        // Begin neighbor analysys
        ////////////////////////////////////////////////////////////////////////
        let mut best_fire: Option<Point<float::T>> = None;
        if in_bounds {
            for neighbor in iter_neighbors(&geo_ref, shared) {
                // neighbor has fire
                let fire: Option<FireSimple> = (*shared.wrapping_add(shared_ix)).map_or(
                    {
                        let terrain = TerrainCuda {
                            d1hr: d1hr[global_ix],
                            d10hr: d10hr[global_ix],
                            d100hr: d100hr[global_ix],
                            herb: herb[global_ix],
                            wood: wood[global_ix],
                            wind_speed: wind_speed[global_ix],
                            wind_azimuth: wind_azimuth[global_ix],
                            slope: slope[global_ix],
                            aspect: aspect[global_ix],
                        };
                        Catalog::STANDARD.burn_simple(model[global_ix], &terrain.into())
                    },
                    |p| p.fire,
                );
                let neighbor_as_reference = || match neighbor.as_reference() {
                    Some(reference) => {
                        let time = reference.time_to(&geo_ref, pos);
                        when_lt_max_time(Point {
                            time,
                            fire,
                            reference: Some(reference),
                        })
                    }
                    None => None,
                };
                let reference = (|| {
                    let neigh_ref = neighbor.point.effective_ref(pos)?;
                    let pos = Coord {
                        x: pos.x as float::T,
                        y: pos.y as float::T,
                    };
                    let other = Coord {
                        x: neigh_ref.pos.x as float::T,
                        y: neigh_ref.pos.y as float::T,
                    };
                    let possible_blockage_pos = geometry::line_to(pos, other).nth(1)?;
                    let shared_blockage_ix = {
                        if possible_blockage_pos.x >= 0.0
                            && possible_blockage_pos.x < BLOCK_WIDTH as float::T
                            && possible_blockage_pos.y >= 0.0
                            && possible_blockage_pos.y < BLOCK_HEIGHT as float::T
                        {
                            Some(
                                possible_blockage_pos.x as usize
                                    + possible_blockage_pos.y as usize * BLOCK_WIDTH,
                            )
                        } else {
                            None
                        }
                    }?;
                    let possible_blockage = (*shared.wrapping_add(shared_blockage_ix))?;
                    let blockage_fire = possible_blockage.fire?;
                    if similar_fires(&blockage_fire, &neighbor.point.fire?) {
                        Some(neigh_ref)
                    } else {
                        None
                    }
                })();
                let point = match (fire, reference) {
                    // We are combustible and reference can be used
                    (Some(fire), Some(reference)) => {
                        if similar_fires(&fire, &reference.fire) {
                            let time = reference.time_to(&geo_ref, pos);
                            when_lt_max_time(Point {
                                time,
                                fire: Some(fire),
                                reference: Some(reference),
                            })
                        } else {
                            // Reference is not valid, use the neighbor
                            neighbor_as_reference()
                        }
                    }
                    // We are combustible but reference is not valid, use the neighbor
                    (Some(_), None) => neighbor_as_reference(),
                    // We are not combustible but reference can be used.
                    // We assign an acces time but a None fire
                    (None, Some(reference)) => {
                        let time = reference.time_to(&geo_ref, pos);
                        when_lt_max_time(Point {
                            time,
                            fire: None,
                            reference: Some(reference),
                        })
                    }
                    // Not combustible and invalid reference
                    (None, None) => None,
                };
                // Update the best_fire with the one with closes access time
                best_fire = match (point, best_fire) {
                    (Some(point), Some(best_fire)) if point.time < best_fire.time => Some(point),
                    _ => best_fire,
                };
            }
        }
        ///////////////////////////////////////////////////
        // End of neighbor analysys, save point if improves
        ///////////////////////////////////////////////////
        let improved = if let Some(point) = best_fire {
            point.save(
                global_ix,
                time,
                speed_max,
                azimuth_max,
                eccentricity,
                refs_x,
                refs_y,
            );
            1
        } else {
            0
        };
        // If no threads in this block have improved
        // break the block loop
        if thread::sync_threads_or(improved) == 0 {
            break;
        }
    }
}

// TODO: Fine-tune these constants and make them configurable
fn similar_fires(a: &FireSimple, b: &FireSimple) -> bool {
    (a.speed_max - b.speed_max).abs().get::<meter_per_second>() < 0.5
        && (a.azimuth_max - b.azimuth_max).abs().get::<degree>() < 5.0
        && (a.eccentricity - b.eccentricity).abs().get::<ratio>() < 0.1
}

impl<T: CoordFloat> PointRef<T> {
    fn time_to(&self, geo_ref: &GeoReference<f64>, to: Coord<usize>) -> float::T {
        let from_pos = Coord {
            x: self.pos.x as _,
            y: self.pos.y as _,
        };
        let to = Coord {
            x: to.x as _,
            y: to.y as _,
        };
        let bearing = Angle::new::<radian>(geo_ref.bearing(from_pos, to) as _);
        let speed = self.fire.spread(bearing).speed().get::<meter_per_second>();
        let distance: float::T =
            NumCast::from(geo_ref.distance(from_pos, to)).expect("conversion failed");
        let time = <float::T as NumCast>::from(self.time).expect("conversion failed");
        time + (distance / speed)
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct PointRef<T>
where
    T: CoordFloat,
{
    time: T,
    pos: Coord<usize>,
    fire: FireSimple,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct Point<T>
where
    T: CoordFloat,
{
    time: T,
    fire: Option<FireSimple>,
    reference: Option<PointRef<T>>,
}

impl Point<float::T> {
    fn effective_ref(&self, my_pos: Coord<usize>) -> Option<PointRef<float::T>> {
        self.reference.or(Some(PointRef {
            pos: my_pos,
            time: self.time,
            fire: self.fire?,
        }))
    }
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
        let fire = Some(FireSimpleCuda {
            azimuth_max: (*azimuth_max.wrapping_add(idx))?,
            speed_max: (*speed_max.wrapping_add(idx))?,
            eccentricity: (*eccentricity.wrapping_add(idx))?,
        })
        .map(|f| f.into());
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
            })
            .expect("reference is not combustible")
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

    unsafe fn save(
        &self,
        idx: usize,
        time: *mut Option<float::T>,
        speed_max: *mut Option<float::T>,
        azimuth_max: *mut Option<float::T>,
        eccentricity: *mut Option<float::T>,
        ref_x: *mut Option<usize>,
        ref_y: *mut Option<usize>,
    ) {
        *time.wrapping_add(idx) = Some(self.time);
        if let Some(fire) = self.fire {
            let fire = Into::<FireSimpleCuda>::into(fire);
            *speed_max.wrapping_add(idx) = Some(fire.speed_max);
            *azimuth_max.wrapping_add(idx) = Some(fire.azimuth_max);
            *eccentricity.wrapping_add(idx) = Some(fire.eccentricity);
        } else {
            *speed_max.wrapping_add(idx) = None;
            *azimuth_max.wrapping_add(idx) = None;
            *eccentricity.wrapping_add(idx) = None;
        }
        if let Some(reference) = self.reference {
            *ref_x.wrapping_add(idx) = Some(reference.pos.x);
            *ref_y.wrapping_add(idx) = Some(reference.pos.y);
        } else {
            *ref_x.wrapping_add(idx) = None;
            *ref_y.wrapping_add(idx) = None;
        }
    }
}

struct Neighbor {
    global_pos: Coord<usize>,
    local_pos: Coord<usize>,
    point: Point<float::T>,
}

impl Neighbor {
    fn as_reference(&self) -> Option<PointRef<float::T>> {
        Some(PointRef {
            time: self.point.time,
            pos: self.global_pos,
            fire: self.point.fire?,
        })
    }
}

unsafe fn iter_neighbors(
    geo_ref: &GeoReference<f64>,
    shared: *const Option<Point<float::T>>,
) -> impl Iterator<Item = Neighbor> {
    // Dimensions of the total area
    let width = geo_ref.size[0] as u32;
    let height = geo_ref.size[1] as u32;
    // Index of this thread into total area
    let idx_2d = thread::index_2d();
    (-1..2)
        .map(|dj| (-1..2).map(move |di| (di, dj)))
        .flatten()
        .filter(|(di, dj)| *di == 0 && *dj == 0)
        .filter_map(move |(di, dj)| {
            let local_pos = Coord {
                x: thread::thread_idx_x() as i32 + di,
                y: thread::thread_idx_y() as i32 + dj,
            };
            // Neighbor position in global pixel coords
            let global_pos = Coord {
                x: idx_2d.x as i32 + di,
                y: idx_2d.y as i32 + dj,
            };
            // If this neighbor is outside this block or global area
            // continue
            if local_pos.x < 0
                || local_pos.x >= BLOCK_WIDTH as _
                || local_pos.y < 0
                || local_pos.y >= BLOCK_HEIGHT as _
                || global_pos.x < 0
                || global_pos.x >= width as _
                || global_pos.y < 0
                || global_pos.y >= height as _
            {
                None
            } else {
                let local_pos = Coord {
                    x: local_pos.x as usize,
                    y: local_pos.y as usize,
                };
                let global_pos = Coord {
                    x: global_pos.x as usize,
                    y: global_pos.y as usize,
                };
                let shared_mem_ix = local_pos.x + local_pos.y * BLOCK_WIDTH;
                let point = (*shared.wrapping_add(shared_mem_ix))?;
                Some(Neighbor {
                    local_pos,
                    global_pos,
                    point,
                })
            }
        })
}
