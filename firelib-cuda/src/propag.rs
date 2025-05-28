use core::ops::Div;
use core::ptr::NonNull;
use glam::f32::*;
use glam::i32::*;
use glam::usize::*;
use cuda_std::thread::*;
use core::sync::atomic::Ordering;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::Angle;
use firelib_rs::*;
use geometry::{GeoReference};
use uom::num_traits::NumCast;
use uom::si::angle::{degree, radian};
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

const BLOCK_WIDTH: usize = 16;
const BLOCK_HEIGHT: usize = 16;
const SHARED_SIZE: usize = BLOCK_WIDTH * BLOCK_HEIGHT;

unsafe fn read_volatile<T: Copy>(p: *const T) -> T {
    *p
    //core::intrinsics::volatile_load(p)
}
unsafe fn write_volatile<T: Copy>(p: *mut T, v: T) {
    *p = v;
    //core::intrinsics::volatile_store(p,v)
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn propag(
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
    progress: *mut float::T,
    geo_ref: GeoReference,
    num_blocks: u32,
    max_time: float::T,
) {
    // Arrays in shared memory for fast analysis within block
    let shared : *mut Option<Point> =
        shared_array![Option<Point>; SHARED_SIZE];

    let when_lt_max_time = |p: Point| if p.time < max_time { Some(p) } else { None };

    // Dimensions of the total area
    let width = geo_ref.size[0] as u32;
    let height = geo_ref.size[1] as u32;

    // Index of this thread into total area
    let idx_2d = thread::index_2d();
    let in_bounds = idx_2d.x < width && idx_2d.y < height;
    let blk_idx = (thread::block_idx_x() + thread::block_idx_y() * thread::grid_dim_x()) as usize;

    // Our position in global pixel coords
    let pos = USizeVec2 {
        x: idx_2d.x as usize,
        y: idx_2d.y as usize,
    };

    // Our global index
    let global_ix = (idx_2d.x + idx_2d.y * width) as usize;

    // Our index into the shared area
    let shared_ix = thread::thread_idx_x() as usize + thread::thread_idx_y() as usize * BLOCK_WIDTH;

    ////////////////////////////////////////////////////////////////////////
    // First phase, load data from global to shared memory
    ////////////////////////////////////////////////////////////////////////
    // are we in-bounds of the target raster?
    let (fire, me) = if in_bounds {
        let (fire, me) = {
            //first_iteration = false;
            // load initial fire
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
            let fire : Option<FireSimpleCuda>=
                Catalog::STANDARD.burn_simple(model[global_ix], &terrain.into()).map(|f|f.into());
            if fire.is_some() {
                save_fire(fire, global_ix, speed_max, azimuth_max, eccentricity);
            }
            (fire, Point::load(
                pos,
                global_ix,
                time,
                speed_max,
                azimuth_max,
                eccentricity,
                refs_x,
                refs_y,
            ))
        };
        (fire, me)
    } else {
        (None, None)
    };
    *shared.add(shared_ix) = me;
    //println!("{}, {}", fire.is_some(), me.is_some());
    thread::sync_threads();

    let mut best_fire: Option<Point> = None;

    ////////////////////////////////////////////////////////////////////////
    // Begin neighbor analysys
    ////////////////////////////////////////////////////////////////////////
    if in_bounds {
        for neighbor in iter_neighbors(&geo_ref, shared) {
            self::println!("vecino con fuego! {:?} {:?}", pos, neighbor.pos);
            let reference = (|| {
                let neigh_ref = neighbor.reference();
                if neigh_ref.pos() == neighbor.pos {
                    return Some(neigh_ref)
                }
                let pos = Vec2 {
                    x: pos.x as float::T,
                    y: pos.y as float::T,
                };
                let other = Vec2 {
                    x: neigh_ref.pos().x as float::T,
                    y: neigh_ref.pos().y as float::T,
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
                let possible_blockage = (*shared.add(shared_blockage_ix))?;
                let blockage_fire = possible_blockage.fire?;
                if similar_fires(blockage_fire, neighbor.point.fire?) {
                    Some(neigh_ref)
                } else {
                    None
                }
            })();
            let point = match (fire, reference) {
                // We are combustible and reference can be used
                (Some(fire), Some(reference)) => {
                    if similar_fires(fire, reference.fire) {
                        let time = reference.time_to(&geo_ref, pos);
                        //self::println!("case 1.1 {}", time>0.0 && time<max_time);
                        when_lt_max_time(Point {
                            time,
                            fire: Some(fire),
                            reference: reference,
                        })
                    } else {
                        //self::println!("case 1.2");
                        // Reference is not valid, use the neighbor
                        match neighbor.as_reference() {
                            Some(reference) if reference.pos() != pos => {
                                //self::println!("case 1.2.1");
                                let time = reference.time_to(&geo_ref, pos);
                                when_lt_max_time(Point {
                                    time,
                                    fire: Some(fire),
                                    reference
                                })
                            }
                            _ => None,
                        }
                    }
                }
                // We are not combustible but reference can be used.
                // We assign an access time but a None fire
                (None, Some(reference)) => {
                    //self::println!("case 3");
                    let time = reference.time_to(&geo_ref, pos);
                    when_lt_max_time(Point {
                        time,
                        fire: None,
                        reference,
                    })
                }
                // We are combustible but reference is not valid
                (Some(_), None) => {
                    None
                },
                // Not combustible and invalid reference
                (None, None) => None,
            };
            // Update the best_fire with the one with lowest access time
            //self::println!("lelo: {:?}, {:?}", point, point);
            best_fire = match (point, best_fire) {
                (Some(point), Some(best_fire)) if point.time < best_fire.time => {
                    Some(point)
                },
                (Some(point), None)  => {
                    Some(point)
                },
                _ => best_fire,
            };
        };
        ///////////////////////////////////////////////////
        // End of neighbor analysys, save point if improves
        ///////////////////////////////////////////////////
        if let Some(point) = best_fire {
            self::println!("saving");
            point.save(
                global_ix,
                time,
                speed_max,
                azimuth_max,
                eccentricity,
                refs_x,
                refs_y,
            );
        }
    }
}

// TODO: Fine-tune these constants and make them configurable
fn similar_fires(a: FireSimpleCuda, b: FireSimpleCuda) -> bool {
    (a.speed_max - b.speed_max).abs() < 1.0
        && (a.azimuth_max - b.azimuth_max).abs() < 5.0 / (2.0*core::f32::consts::PI)
        && (a.eccentricity - b.eccentricity).abs() < 0.1
}

impl PointRef {
    fn time_to(&self, geo_ref: &GeoReference, to: USizeVec2) -> f32 {
        let from_pos = Vec2 {
            x: self.pos().x as _,
            y: self.pos().y as _,
        };
        let to = Vec2 {
            x: to.x as _,
            y: to.y as _,
        };
        let bearing = Angle::new::<radian>(geo_ref.bearing(from_pos, to));
        let speed = self.fire.spread(bearing).speed().get::<meter_per_second>();
        let distance = geo_ref.distance(from_pos, to);
        let time = self.time;
        time + (distance / speed)
    }
}

#[derive(Copy, Clone, PartialEq, Debug, cust_core::DeviceCopy)]
#[repr(C, align(16))]
struct PointRef
{
    time: f32,
    pos_x: usize,
    pos_y: usize,
    fire: FireSimpleCuda,
}

impl PointRef {
    fn pos(&self) -> USizeVec2 {
        USizeVec2 {x : self.pos_x, y: self.pos_y }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, cust_core::DeviceCopy)]
#[repr(C, align(16))]
struct Point
{
    time: f32,
    fire: Option<FireSimpleCuda>,
    reference: PointRef,
}

impl Point {
    unsafe fn load(
        pos: USizeVec2,
        idx: usize,
        time: *const Option<float::T>,
        speed_max: *const Option<float::T>,
        azimuth_max: *const Option<float::T>,
        eccentricity: *const Option<float::T>,
        ref_x: *const Option<usize>,
        ref_y: *const Option<usize>,
    ) -> Option<Self> {
        let p_time = read_volatile(time.add(idx))?;
        let fire = load_fire(idx, speed_max, azimuth_max, eccentricity);
        let ref_pos = USizeVec2 {
            x: read_volatile(ref_x.add(idx))?,
            y: read_volatile(ref_y.add(idx))?
        };
        let reference = match (fire) {
            (Some(_)) if ref_pos != pos => {
                let ref_ix: usize = ref_pos.x + ref_pos.y * BLOCK_WIDTH;
                self::println!("fuego tiemposo! {:?} {:?} {:?}", pos, ref_pos, read_volatile(time.add(ref_ix)).is_some());
                let fire = load_fire(ref_ix, speed_max, azimuth_max, eccentricity)?;
                self::println!("fuego tiemposo1");
                let time = read_volatile(time.add(ref_ix))?;
                self::println!("fuego tiemposo2");
                Some(PointRef { time, pos_x: ref_pos.x, pos_y:ref_pos.y, fire, })
            },
            (Some(fire)) =>
                Some(PointRef {
                    time: p_time,
                    pos_x: pos.x,
                    pos_y: pos.y,
                    fire,
                }),
            _ => None
        };
        self::println!("fuego tiempo! {:?} {:?}", read_volatile(time.add(idx)).is_some(), reference.is_some());
        Some(Point {time: p_time, fire, reference:reference? })
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
        write_volatile(time.add(idx), Some(self.time));
        save_fire(self.fire, idx, speed_max, azimuth_max, eccentricity);
        let reference = self.reference;
        write_volatile(ref_x.add(idx), Some(reference.pos_x));
        write_volatile(ref_y.add(idx), Some(reference.pos_y));
    }
    fn as_reference(&self, pos: USizeVec2) -> Option<PointRef> {
        Some(PointRef {
            time: self.time,
            pos_x:pos.x,
            pos_y:pos.y,
            fire: self.fire?,
        })
    }
}

unsafe fn load_fire(
    idx: usize,
    speed_max: *const Option<float::T>,
    azimuth_max: *const Option<float::T>,
    eccentricity: *const Option<float::T>,
    ) -> Option<FireSimpleCuda> {
  let azimuth_max = read_volatile(azimuth_max.add(idx))?;
  let speed_max = read_volatile(speed_max.add(idx))?;
  let eccentricity = read_volatile(eccentricity.add(idx))?;
  Some(
    (FireSimpleCuda { azimuth_max, speed_max, eccentricity}).into()
  )
}
unsafe fn save_fire(
    fire: Option<FireSimpleCuda>,
    idx: usize,
    speed_max: *mut Option<float::T>,
    azimuth_max: *mut Option<float::T>,
    eccentricity: *mut Option<float::T>,
    ) {
    match fire {
        Some(fire) => {
            let fire = Into::<FireSimpleCuda>::into(fire);
            write_volatile(speed_max.add(idx), Some(fire.speed_max));
            write_volatile(azimuth_max.add(idx), Some(fire.azimuth_max));
            write_volatile(eccentricity.add(idx), Some(fire.eccentricity));
        },
        None => {
            write_volatile(speed_max.add(idx), None);
            write_volatile(azimuth_max.add(idx), None);
            write_volatile(eccentricity.add(idx), None);
        }
    }
}

#[derive(Debug)]
#[repr(C, align(16))]
struct Neighbor {
    point: Point,
    pos: USizeVec2,
}

impl Neighbor {
    fn as_reference(&self) -> Option<PointRef> {
        self.point.as_reference(self.pos)
    }
    fn reference(&self) -> PointRef {
        self.point.reference
    }
}

unsafe fn iter_neighbors(
    geo_ref: &GeoReference,
    shared: *const Option<Point>,
) -> impl Iterator<Item = Neighbor> {
    // Dimensions of the total area
    let width = geo_ref.size[0] as u32;
    let height = geo_ref.size[1] as u32;
    // Index of this thread into total area
    let idx_2d = thread::index_2d();
    (-1..2)
        .map(|dj| (-1..2).map(move |di| (di, dj)))
        .flatten()
        .filter(|(di, dj)| !(*di == 0 && *dj == 0))
        .filter_map(move |(di, dj)| {
            let local_pos = IVec2 {
                x: thread::thread_idx_x() as i32 + di,
                y: thread::thread_idx_y() as i32 + dj,
            };
            // Neighbor position in global pixel coords
            let pos = IVec2 {
                x: idx_2d.x as i32 + di,
                y: idx_2d.y as i32 + dj,
            };
            // If this neighbor is outside this block or global area
            // continue
            if local_pos.x < 0
                || local_pos.x >= BLOCK_WIDTH as _
                || local_pos.y < 0
                || local_pos.y >= BLOCK_HEIGHT as _
                || pos.x < 0
                || pos.x >= width as _
                || pos.y < 0
                || pos.y >= height as _
            {
                None
            } else {
                let local_pos = USizeVec2 {
                    x: local_pos.x as usize,
                    y: local_pos.y as usize,
                };
                let pos = USizeVec2 {
                    x: pos.x as usize,
                    y: pos.y as usize,
                };
                let shared_mem_ix = local_pos.x + local_pos.y * BLOCK_WIDTH;
                let point = (*shared.add(shared_mem_ix))?;
                Some(Neighbor {
                    pos,
                    point,
                })
            }
        })
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn pre_burn(
    model: &[Option<usize>],
    d1hr: &[Option<float::T>],
    d10hr: &[Option<float::T>],
    d100hr: &[Option<float::T>],
    herb: &[Option<float::T>],
    wood: &[Option<float::T>],
    wind_speed: &[Option<float::T>],
    wind_azimuth: &[Option<float::T>],
    slope: &[Option<float::T>],
    aspect: &[Option<float::T>],
    speed_max: *mut Option<float::T>,
    azimuth_max: *mut Option<float::T>,
    eccentricity: *mut Option<float::T>,
) {
    let i = thread::index_1d() as usize;
    if i < model.len() {
        let terrainModel = (|| Some((TerrainCuda {
            d1hr: d1hr[i]?,
            d10hr: d10hr[i]?,
            d100hr: d100hr[i]?,
            herb: herb[i]?,
            wood: wood[i]?,
            wind_speed: wind_speed[i]?,
            wind_azimuth: wind_azimuth[i]?,
            slope: slope[i]?,
            aspect: aspect[i]?,
        }, model[i]?)))();
        if let Some(fire) = terrainModel.and_then(|(terrain, model)|
            Catalog::STANDARD.burn_simple(model, &terrain.into()))
        {
            let fire = Into::<FireSimpleCuda>::into(fire);
            *speed_max.wrapping_add(i) = Some(fire.speed_max);
            *azimuth_max.wrapping_add(i) = Some(fire.azimuth_max);
            *eccentricity.wrapping_add(i) = Some(fire.eccentricity);
        } else {
            *speed_max.wrapping_add(i) = None;
            *azimuth_max.wrapping_add(i) = None;
            *eccentricity.wrapping_add(i) = None;
        }
    }
}
