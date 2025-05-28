use core::ops::Div;
use core::ptr::NonNull;
use firelib_rs::float::*;
use core::sync::atomic::Ordering;
use cuda_std::shared::dynamic_shared_mem;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::*;
use geometry::GeoReference;
use glam::f32::*;
use glam::i32::*;
use glam::usize::*;
use min_max_traits::Max;
use uom::num_traits::NumCast;
use uom::si::angle::{degree, radian};
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

pub const HALO_RADIUS: usize = 4;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn propag(
    settings: Settings,
    speed_max: &[float::T],
    azimuth_max: &[float::T],
    eccentricity: &[float::T],
    time: *mut f32,
    refs_x: *mut usize,
    refs_y: *mut usize,
    progress: *mut u32,
) {
    // Arrays in shared memory for fast analysis within block
    let shared: *mut Point = dynamic_shared_mem();
    let Settings { geo_ref, max_time } = settings;
    let when_lt_max_time = |p: Point| if p.time < max_time { Some(p) } else { None };

    // Dimensions of the total area
    let GeoReference { width, height, .. } = geo_ref;
    // Index of this thread into total area
    let idx_2d = thread::index_2d();
    // Our position in global pixel coords
    let pos = USizeVec2 {
        x: idx_2d.x as usize,
        y: idx_2d.y as usize,
    };
    // Our global index
    let global_ix = (idx_2d.x + idx_2d.y * width) as usize;
    let in_bounds = idx_2d.x < width && idx_2d.y < height;

    // mark no improvement
    let block_ix = (thread::block_idx_x() + thread::block_idx_y() * thread::grid_dim_x()) as usize;
    if !block_ix < (thread::grid_dim_x() * thread::grid_dim_y()) as usize {
        panic!();
    };
    write_volatile(progress.add(block_ix), 0);

    let to_shared_off = |x, y| {
        let pos = IVec2 {
            x: pos.x as _,
            y: pos.y as _,
        };
        let pos = pos + IVec2 { x, y };
        if pos.x >= 0 && pos.y >= 0 && pos.x < width as i32 && pos.y < height as i32 {
            let pos = USizeVec2 {
                x: pos.x as _,
                y: pos.y as _,
            };
            Some(compute_shared_ix(&pos))
        } else {
            None
        }
    };
    let to_global_off = |ox, oy| {
        let x = idx_2d.x as i32 + ox;
        let y = idx_2d.y as i32 + oy;
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            Some((x + y * width as i32) as usize)
        } else {
            None
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // First phase, load data from global to shared memory
    ////////////////////////////////////////////////////////////////////////
    let preload_point = |x, y| {
        let shared_off = to_shared_off(x, y);
        let global_off = to_global_off(x, y);
        match (shared_off, global_off) {
            (Some(shared_off), Some(global_off)) => {
                let p = Point::load(
                    &geo_ref,
                    pos,
                    global_off,
                    speed_max,
                    azimuth_max,
                    eccentricity,
                    time,
                    refs_x,
                    refs_y,
                );
                write_shared(shared.add(shared_off), p.unwrap_or(Point::NULL));
                p
            }
            (Some(shared_off), None) => {
                write_shared(shared.add(shared_off), Point::NULL);
                None
            }
            (None, Some(_)) => {
                /*
                println!("{:?} {:?} {:?} {:?}", x, y, shared_off, global_off);
                */
                panic!("should not happen");
            }
            (None, None) => None,
        }
    };
    // are we in-bounds of the target raster?
    let (is_new, fire) = if in_bounds {
        let is_new = preload_point(0, 0).is_none();
        let fire = if is_new {
            load_fire(global_ix, speed_max, azimuth_max, eccentricity)
        } else {
            None
        };
        // Preload boundaries
        let radius = HALO_RADIUS as i32;
        let x_near_border = (thread::thread_idx_x() as i32) < radius;
        let y_near_border = (thread::thread_idx_y() as i32) < radius;
        if x_near_border {
            let _ = preload_point(-radius, 0);
            let _ = preload_point((thread::block_dim_x() as usize) as i32, 0);
        }
        if y_near_border {
            let _ = preload_point(0, -radius);
            let _ = preload_point(0, (thread::block_dim_y() as usize) as i32);
        }
        if x_near_border && y_near_border {
            let _ = preload_point(-radius, -radius);
            let _ = preload_point(
                (thread::block_dim_x() as usize) as i32,
                (thread::block_dim_y() as usize) as i32,
            );
        }
        (is_new, fire)
    } else {
        (false, None)
    };

    thread::sync_threads();

    ////////////////////////////////////////////////////////////////////////
    // Begin neighbor analysys
    ////////////////////////////////////////////////////////////////////////
    let mut best_fire: Option<Point> = None;
    let improved = if is_new && in_bounds {
        for neighbor in iter_neighbors(&geo_ref, shared) {
            let reference = (|| {
                let candidate = neighbor.reference();
                if candidate.pos() == pos {
                    return None;
                };
                let pos = IVec2 {
                    x: pos.x as _,
                    y: pos.y as _,
                };
                let npos = IVec2 {
                    x: neighbor.pos.x as _,
                    y: neighbor.pos.y as _,
                };
                let candidate_pos = IVec2 {
                    x: candidate.pos().x as _,
                    y: candidate.pos().y as _,
                };
                let possible_blockage_pos = geometry::neighbor_in_direction(pos, candidate_pos);
                if possible_blockage_pos != pos {
                    let possible_blockage_pos = USizeVec2 {
                        x: possible_blockage_pos.x as _,
                        y: possible_blockage_pos.y as _,
                    };
                    let shared_blockage_ix = compute_shared_ix(&possible_blockage_pos);
                    let possible_blockage = read_shared(shared.add(shared_blockage_ix));
                    if possible_blockage != Point::NULL
                        && possible_blockage.fire != FireSimpleCuda::NULL
                        && similar_fires(possible_blockage.fire, candidate.fire)
                    {
                        Some(candidate)
                    } else {
                        None
                    }
                } else {
                    Some(candidate)
                }
            })();
            //self::println!("vecino con fuego! {:?} {:?} {:?} {:?}", pos, me.is_some(), reference.is_some(), neighbor.reference().pos());
            let point = (|| match (fire, reference) {
                // We are combustible and reference can be used
                (Some(fire), Some(reference)) => {
                    if similar_fires(fire, reference.fire) {
                        let time = reference.time_to(&geo_ref, pos)?;
                        //self::println!("case 1.1 {:?} {:?}", pos, reference.pos());
                        when_lt_max_time(Point {
                            time,
                            fire,
                            reference,
                        })
                    } else {
                        //self::println!("case 1.2");
                        //panic!("caca"); //FIXME
                        // Reference is not valid, use the neighbor
                        match neighbor.as_reference() {
                            Some(reference)
                                if reference.pos() != pos
                                    && similar_fires(fire, reference.fire) =>
                            {
                                //self::println!("case 1.2.1");
                                let time = reference.time_to(&geo_ref, pos)?;
                                when_lt_max_time(Point {
                                    time,
                                    fire,
                                    reference,
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
                    //panic!("caca"); //FIXME
                    let time = reference.time_to(&geo_ref, pos)?;
                    when_lt_max_time(Point {
                        time,
                        fire: FireSimpleCuda::NULL,
                        reference,
                    })
                }
                // Not combustible or invalid reference
                _ => None,
            })();
            // Update the best_fire with the one with lowest access time
            best_fire = match (point, best_fire) {
                (Some(point), Some(best_fire)) if point.time < best_fire.time => Some(point),
                (Some(point), None) => Some(point),
                _ => best_fire,
            };
        }
        ///////////////////////////////////////////////////
        // End of neighbor analysys, save point if improves
        ///////////////////////////////////////////////////
        match best_fire {
            /*
            (Some(me), Some(best_fire)) if best_fire.time < me.time => {
                best_fire.save(global_ix, time, refs_x, refs_y);
                1
            }
            (Some(me), _) => {
                me.save(global_ix, time, refs_x, refs_y);
                1
            },
            */
            Some(best_fire) => {
                write_shared(shared.add(compute_shared_ix(&pos)), best_fire);
                best_fire.save(global_ix, time, refs_x, refs_y);
                1
            }
            _ => 0,
        }
    } else {
        0
    }; // in_bounds
    let any_improved = thread::sync_threads_or(improved) > 0;
    if any_improved && thread::thread_idx_x() == 0 && thread::thread_idx_y() == 0 {
        let block_ix =
            (thread::block_idx_x() + thread::block_idx_y() * thread::grid_dim_x()) as usize;
        //println!("progress {} {}", thread::block_idx_x(), thread::block_idx_y());
        write_volatile(progress.add(block_ix), 1);
    }
}

// TODO: Fine-tune these constants and make them configurable
fn similar_fires(a: FireSimpleCuda, b: FireSimpleCuda) -> bool {
    (a.speed_max - b.speed_max).abs() < 1.0
        && (a.azimuth_max - b.azimuth_max).abs() < 5.0 / (2.0 * core::f32::consts::PI)
        && (a.eccentricity - b.eccentricity).abs() < 0.1
}

fn compute_shared_ix(pos: &USizeVec2) -> usize {
    let shared_x = (pos.x % (thread::block_dim_x() as usize)) + HALO_RADIUS;
    let shared_y = (pos.y % (thread::block_dim_y() as usize)) + HALO_RADIUS;
    let ix = shared_x + shared_y * ((thread::block_dim_x() as usize) + HALO_RADIUS * 2);
    /*
    assert!(ix < ((16+HALO_RADIUS*2)*(16+HALO_RADIUS*2)));
    assert!(shared_x < (thread::block_dim_x() as usize) + HALO_RADIUS * 2
        && shared_y < (thread::block_dim_y() as usize) + HALO_RADIUS * 2);
    */
    ix
}

impl PointRef {
    pub const NULL: Self = Self {
        time: 0.0,
        pos_x: Max::MAX,
        pos_y: Max::MAX,
        fire: FireSimpleCuda::NULL,
    };

    fn time_to(&self, geo_ref: &GeoReference, to: USizeVec2) -> Option<f32> {
        let from_pos = Vec2 {
            x: self.pos().x as _,
            y: self.pos().y as _,
        };
        let to = Vec2 {
            x: to.x as _,
            y: to.y as _,
        };
        let bearing = Angle::new::<radian>(geo_ref.bearing(from_pos, to));
        let fire : FireSimple = self.fire.into();
        let speed = fire.spread(bearing).speed().get::<meter_per_second>();
        let distance = geo_ref.distance(from_pos, to);
        let time = self.time;
        if speed > 1e-6 {
            Some((time + (distance / speed)))
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct PointRef {
    pub time: f32,
    pub pos_x: usize,
    pub pos_y: usize,
    pub fire: FireSimpleCuda,
}

impl PointRef {
    pub fn pos(&self) -> USizeVec2 {
        USizeVec2 {
            x: self.pos_x,
            y: self.pos_y,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, cust_core::DeviceCopy)]
#[repr(C)]
pub struct Point {
    pub time: f32,
    pub fire: FireSimpleCuda,
    pub reference: PointRef,
}


impl Point {
    pub const NULL: Self = Self {
        time: 0.0,
        fire: FireSimpleCuda::NULL,
        reference: PointRef::NULL,
    };

    unsafe fn load(
        geo_ref: &GeoReference,
        pos: USizeVec2,
        idx: usize,
        speed_max: &[float::T],
        azimuth_max: &[float::T],
        eccentricity: &[float::T],
        time: *const f32,
        ref_x: *const usize,
        ref_y: *const usize,
    ) -> Option<Self> {
        let p_time = read_volatile(time.add(idx));
        if p_time == Max::MAX {
            None
        } else {
            let fire = load_fire(idx, speed_max, azimuth_max, eccentricity)
                .unwrap_or(FireSimpleCuda::NULL);
            let reference = {
                let ref_x = read_volatile(ref_x.add(idx));
                let ref_y = read_volatile(ref_y.add(idx));
                if ref_x == Max::MAX || ref_y == Max::MAX {
                    panic!();
                };
                let ref_pos = USizeVec2 { x: ref_x, y: ref_y };
                let ref_ix: usize = ref_pos.x + ref_pos.y * geo_ref.width as usize;
                let fire =
                    load_fire(ref_ix, speed_max, azimuth_max, eccentricity).expect("can't be NULL");
                let r_time = read_volatile(time.add(ref_ix));
                if r_time == Max::MAX {
                    panic!();
                };
                PointRef {
                    time: r_time,
                    pos_x: ref_pos.x,
                    pos_y: ref_pos.y,
                    fire,
                }
            };
            Some(Point {
                time: p_time,
                fire,
                reference,
            })
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern fn save(&self, idx: usize, time: *mut f32, ref_x: *mut usize, ref_y: *mut usize) {
        write_volatile(time.add(idx), self.time);
        write_volatile(ref_x.add(idx), self.reference.pos_x);
        write_volatile(ref_y.add(idx), self.reference.pos_y);
    }

    fn as_reference(&self, pos: USizeVec2) -> Option<PointRef> {
        if self.fire != FireSimpleCuda::NULL {
            Some(PointRef {
                time: self.time,
                pos_x: pos.x,
                pos_y: pos.y,
                fire: self.fire,
            })
        } else {
            None
        }
    }
}

fn load_fire(
    idx: usize,
    speed_max: &[float::T],
    azimuth_max: &[float::T],
    eccentricity: &[float::T],
) -> Option<FireSimpleCuda> {
    let fire = FireSimpleCuda {
        azimuth_max: azimuth_max[idx],
        speed_max: speed_max[idx],
        eccentricity: eccentricity[idx],
    };
    if fire != FireSimpleCuda::NULL {
        Some(fire)
    } else {
        None
    }
}

#[derive(Debug)]
#[repr(C)]
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

#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct Settings {
    pub geo_ref: GeoReference,
    pub max_time: f32,
}

impl Settings {
    #[unsafe(no_mangle)]
    pub extern "C" fn create(geo_ref: GeoReference, max_time: f32) -> Self {
        Self { geo_ref, max_time }
    }
}

unsafe fn iter_neighbors(
    geo_ref: &GeoReference,
    shared: *const Point,
) -> impl Iterator<Item = Neighbor> {
    // Dimensions of the total area
    let GeoReference { width, height, .. } = *geo_ref;
    // Index of this thread into total area
    let idx_2d = thread::index_2d();
    (-1..2)
        .flat_map(|dj| (-1..2).map(move |di| (di, dj)))
        .filter(|(di, dj)| !(*di == 0 && *dj == 0))
        .filter_map(move |(di, dj)| {
            // Neighbor position in global pixel coords
            let pos = IVec2 {
                x: idx_2d.x as i32 + di,
                y: idx_2d.y as i32 + dj,
            };
            // If this neighbor is outside global area continue
            if pos.x < 0 || pos.x >= width as _ || pos.y < 0 || pos.y >= height as _ {
                None
            } else {
                let pos = USizeVec2 {
                    x: pos.x as usize,
                    y: pos.y as usize,
                };
                let shared_mem_ix = compute_shared_ix(&pos);
                let point = read_shared(shared.add(shared_mem_ix));
                Some(Neighbor { pos, point })
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
        let terrainModel = (|| {
            Some((
                TerrainCuda {
                    d1hr: d1hr[i]?,
                    d10hr: d10hr[i]?,
                    d100hr: d100hr[i]?,
                    herb: herb[i]?,
                    wood: wood[i]?,
                    wind_speed: wind_speed[i]?,
                    wind_azimuth: wind_azimuth[i]?,
                    slope: slope[i]?,
                    aspect: aspect[i]?,
                },
                model[i]?,
            ))
        })();
        if let Some(fire) = terrainModel
            .and_then(|(terrain, model)| Catalog::STANDARD.burn_simple(model, &terrain.into()))
        {
            let fire = Into::<FireSimpleCuda>::into(fire);
            *speed_max.add(i) = Some(fire.speed_max);
            *azimuth_max.add(i) = Some(fire.azimuth_max);
            *eccentricity.add(i) = Some(fire.eccentricity);
        } else {
            *speed_max.add(i) = None;
            *azimuth_max.add(i) = None;
            *eccentricity.add(i) = None;
        }
    }
}


#[cfg_attr(not(target_os = "cuda"), derive(StructOfArray), soa_derive(Debug))]
#[derive(Copy, Clone, Debug, PartialEq, cust_core::DeviceCopy)]
#[repr(C)]
pub struct FireSimpleCuda {
    pub speed_max: float::T,
    pub azimuth_max: float::T,
    pub eccentricity: float::T,
}

impl FireSimpleCuda {
    pub const NULL: Self = Self {
        speed_max: 0.0,
        azimuth_max: 0.0,
        eccentricity: 0.0,
    };
}

impl From<FireSimpleCuda> for FireSimple {
    fn from(f: FireSimpleCuda) -> Self {
        Self {
            speed_max: to_quantity!(Velocity, f.speed_max),
            azimuth_max: to_quantity!(Angle, f.azimuth_max),
            eccentricity: to_quantity!(Ratio, f.eccentricity),
        }
    }
}
impl From<FireSimple> for FireSimpleCuda {
    fn from(f: FireSimple) -> Self {
        Self {
            speed_max: from_quantity!(Velocity, &f.speed_max),
            azimuth_max: from_quantity!(Angle, &f.azimuth_max),
            eccentricity: from_quantity!(Ratio, &f.eccentricity),
        }
    }
}
#[cfg(not(target_os = "cuda"))]
impl From<FireSimpleCudaPtr> for FireSimple {
    fn from(f: FireSimpleCudaPtr) -> Self {
        unsafe { f.read().into() }
    }
}
#[cfg(not(target_os = "cuda"))]
impl From<FireSimpleCudaRef<'_>> for FireSimple {
    fn from(f: FireSimpleCudaRef<'_>) -> Self {
        From::<FireSimpleCuda>::from(f.into())
    }
}
/*
impl CanSpread<'_> for FireSimpleCuda {
    fn azimuth_max(&self) -> Angle {
        to_quantity!(Angle, self.azimuth_max)
    }
    fn eccentricity(&self) -> Ratio {
        to_quantity!(Ratio, self.eccentricity)
    }
}

impl<'a> Spread<'a, FireSimpleCuda> {
    pub fn speed(&self) -> Velocity {
        to_quantity!(Velocity, self.fire.speed_max * self.factor)
    }
}
*/

#[inline(always)]
unsafe fn read_volatile<T: Copy>(p: *const T) -> T {
    *p
    //core::intrinsics::volatile_load(p)
}
#[inline(always)]
unsafe fn write_volatile<T: Copy>(p: *mut T, v: T) {
    *p = v
    //core::intrinsics::volatile_store(p, v)
}
#[inline(always)]
unsafe fn read_shared<T: Copy>(p: *const T) -> T {
    *p
    //core::ptr::read_volatile(p)
}
#[inline(always)]
unsafe fn write_shared<T: Copy>(p: *mut T, v: T) {
    *p = v
    //core::ptr::write_volatile(p, v)
}
