use core::ops::Div;
use cuda_std::prelude::*;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::Angle;
use firelib_rs::*;
use geometry::{Coord, CoordFloat, GeoReference};
use uom::si::angle::{degree, radian};

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

    // Dimensions of the total area
    let width = geo_ref.size[0] as u32;
    let height = geo_ref.size[1] as u32;

    // Index of this thread into total area
    let idx_2d = thread::index_2d();

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
        // Wait for threads from previous iteration
        thread::sync_threads();
        if idx_2d.x < width && idx_2d.y < height {
            let me = Point::<float::T>::load(
                idx,
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

        if !(idx_2d.x < width && idx_2d.y < height) {
            continue;
        }
        ////////////////////////////////////////////////////////////////////////
        // Begin neighbor analysys
        ////////////////////////////////////////////////////////////////////////
        let mut best_fire: Option<Point<float::T>> = None;
        for neighbor in iter_neighbors(&geo_ref) {
            if let Some(neigh) = *shared.wrapping_add(neighbor.shared_mem_ix()) {
                // neighbor has fire
                let fire: Option<FireSimple> = (*shared.wrapping_add(shared_ix)).map_or(
                    {
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
                        Catalog::STANDARD.burn_simple(model[idx], &terrain.into())
                    },
                    |p| p.fire,
                );
                let reference = (|| {
                    let neigh_ref = neigh.effective_ref(pos)?;
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
                    if similar_fires(&blockage_fire, &neigh.fire?) {
                        Some(neigh_ref)
                    } else {
                        None
                    }
                })();
                let point = match (fire, reference) {
                    (Some(fire), Some(reference)) => {
                        if similar_fires(&fire, &reference.fire) {
                            let bearing = Angle::new::<radian>(geo_ref.bearing(
                                Coord {
                                    x: reference.pos.x as _,
                                    y: reference.pos.y as _,
                                },
                                Coord {
                                    x: pos.x as _,
                                    y: pos.y as _,
                                },
                            ) as _);
                            let speed = reference.fire.spread(bearing);
                            let time = 0.0;
                            Some(Point {
                                time,
                                fire: Some(fire),
                                reference: Some(reference),
                            })
                        } else {
                            let bearing = Angle::new::<radian>(geo_ref.bearing(
                                Coord {
                                    x: neighbor.global_pos.x as _,
                                    y: neighbor.global_pos.y as _,
                                },
                                Coord {
                                    x: pos.x as _,
                                    y: pos.y as _,
                                },
                            ) as _);
                            match neigh.fire {
                                Some(neigh_fire) => {
                                    let speed = neigh_fire.spread(bearing);
                                    let time = 0.0;
                                    Some(Point {
                                        time,
                                        fire: Some(fire),
                                        reference: None,
                                    })
                                }
                                None => None,
                            }
                        }
                    }
                    (Some(fire), None) => {
                        todo!()
                    }
                    (None, Some(reference)) => {
                        let bearing = Angle::new::<radian>(geo_ref.bearing(
                            Coord {
                                x: reference.pos.x as _,
                                y: reference.pos.y as _,
                            },
                            Coord {
                                x: pos.x as _,
                                y: pos.y as _,
                            },
                        ) as _);
                        let speed = reference.fire.spread(bearing).speed();
                        let time = 0.0;
                        Some(Point {
                            time,
                            fire: None,
                            reference: Some(reference),
                        })
                    }
                    (None, None) => None,
                };
                best_fire = match (point, best_fire) {
                    (Some(point), Some(best_fire)) if point.time < best_fire.time => {
                        Some(point)
                    }
                    _ => best_fire,
                };
            };
        }
        ///////////////////////////////////////////////////
        // End of neighbor analysys, save point if improves
        ///////////////////////////////////////////////////
        if let Some(point) = best_fire {
            point.save(
                idx,
                time,
                speed_max,
                azimuth_max,
                eccentricity,
                refs_x,
                refs_y,
            )
        }
    }
}

fn similar_fires(a: &FireSimple, b: &FireSimple) -> bool {
    todo!()
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
}

impl Neighbor {
    fn shared_mem_ix(&self) -> usize {
        self.local_pos.x + self.local_pos.y * BLOCK_WIDTH
    }
}

fn iter_neighbors(geo_ref: &GeoReference<f64>) -> impl Iterator<Item = Neighbor> {
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
                Some(Neighbor {
                    local_pos,
                    global_pos,
                })
            }
        })
}
