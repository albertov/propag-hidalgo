use core::ops::Div;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::Angle;
use firelib_rs::*;
use geometry::GeoReference;
use glam::f32::*;
use glam::i32::*;
use glam::usize::*;
use uom::num_traits::NumCast;
use uom::si::angle::{degree, radian};
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

const BLOCK_WIDTH: usize = 16;
const BLOCK_HEIGHT: usize = 16;
const SHARED_SIZE: usize = (BLOCK_WIDTH + 2) * (BLOCK_HEIGHT + 2);

unsafe fn read_volatile<T: Copy>(p: *const T) -> T {
    core::intrinsics::volatile_load(p)
}
unsafe fn write_volatile<T: Copy>(p: *mut T, v: T) {
    core::intrinsics::volatile_store(p, v)
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn propag(
    geo_ref: GeoReference,
    max_time: float::T,
    speed_max: &[Option<float::T>],
    azimuth_max: &[Option<float::T>],
    eccentricity: &[Option<float::T>],
    time: *mut Option<float::T>,
    refs_x: *mut Option<usize>,
    refs_y: *mut Option<usize>,
) {
    // Arrays in shared memory for fast analysis within block
    let shared: *mut Option<Point> = shared_array![Option<Point>; SHARED_SIZE];

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
    let to_global_off = |ox, oy| {
        let x = idx_2d.x as i32 + ox;
        let y = idx_2d.y as i32 + oy;
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            Some((x + y * width as i32) as usize)
        } else {
            None
        }
    };

    // Our index into the shared area
    let shared_ix = compute_shared_ix(&pos).expect("pos must have shared ix");
    let to_shared_off = |x, y| {
        let pos = IVec2 {
            x: pos.x as _,
            y: pos.y as _,
        };
        let pos = pos + IVec2 { x, y };
        let pos = USizeVec2 {
            x: pos.x as _,
            y: pos.y as _,
        };
        compute_shared_ix(&pos)
    };

    ////////////////////////////////////////////////////////////////////////
    // First phase, load data from global to shared memory
    ////////////////////////////////////////////////////////////////////////
    // are we in-bounds of the target raster?
    let (fire, me) = if in_bounds {
        let (fire, me) = {
            let fire = load_fire(global_ix, speed_max, azimuth_max, eccentricity);
            (
                fire,
                Point::load(
                    &geo_ref,
                    pos,
                    global_ix,
                    speed_max,
                    azimuth_max,
                    eccentricity,
                    time,
                    refs_x,
                    refs_y,
                ),
            )
        };
        (fire, me)
    } else {
        (None, None)
    };

    // Preload boundaries
    let preload_point = |x, y| {
        let shared_off = to_shared_off(x, y);
        let global_off = to_global_off(x, y);
        match (shared_off, global_off) {
            (Some(shared_off), Some(global_off)) => {
                *shared.add(shared_off) = Point::load(
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
            }
            (Some(shared_off), None) => *shared.add(shared_off) = None,
            _ => (),
        }
    };
    if thread::thread_idx_x() == 0 && thread::thread_idx_y() == 0 {
        preload_point(-1, -1);
    } else if thread::thread_idx_y() == 0 {
        preload_point(0, -1);
    } else if thread::thread_idx_x() as usize == BLOCK_WIDTH - 1 && thread::thread_idx_y() == 0 {
        preload_point(1, -1);
    } else if thread::thread_idx_x() as usize == BLOCK_WIDTH - 1 {
        preload_point(1, 0);
    } else if thread::thread_idx_x() as usize == BLOCK_WIDTH - 1
        && thread::thread_idx_y() as usize == BLOCK_HEIGHT - 1
    {
        preload_point(1, 1);
    } else if thread::thread_idx_y() as usize == BLOCK_HEIGHT - 1 {
        preload_point(0, 1);
    } else if thread::thread_idx_x() == 0 && thread::thread_idx_y() as usize == BLOCK_HEIGHT - 1 {
        preload_point(-1, 1);
    } else if thread::thread_idx_x() == 0 {
        preload_point(-1, 0);
    }
    *shared.add(shared_ix) = me;
    thread::sync_threads();

    let mut best_fire: Option<Point> = None;

    ////////////////////////////////////////////////////////////////////////
    // Begin neighbor analysys
    ////////////////////////////////////////////////////////////////////////
    if in_bounds {
        for neighbor in iter_neighbors(&geo_ref, shared) {
            let reference = (|| {
                let neigh_ref = neighbor.reference();
                if neigh_ref.pos() == neighbor.pos {
                    return Some(neigh_ref);
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
                let possible_blockage_pos = USizeVec2 {
                    x: possible_blockage_pos.x as _,
                    y: possible_blockage_pos.y as _,
                };
                let shared_blockage_ix = compute_shared_ix(&possible_blockage_pos)?;
                let possible_blockage = (*shared.add(shared_blockage_ix))?;
                let blockage_fire = possible_blockage.fire?;
                if similar_fires(blockage_fire, neighbor.point.fire?) {
                    Some(neigh_ref)
                } else {
                    None
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
                            fire: Some(fire),
                            reference: reference,
                        })
                    } else {
                        //self::println!("case 1.2");
                        // Reference is not valid, use the neighbor
                        match neighbor.as_reference() {
                            Some(reference) if reference.pos() != pos => {
                                //self::println!("case 1.2.1");
                                let time = reference.time_to(&geo_ref, pos)?;
                                when_lt_max_time(Point {
                                    time,
                                    fire: Some(fire),
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
                    let time = reference.time_to(&geo_ref, pos)?;
                    when_lt_max_time(Point {
                        time,
                        fire: None,
                        reference,
                    })
                }
                // We are combustible but reference is not valid
                (Some(_), None) => None,
                // Not combustible and invalid reference
                (None, None) => None,
            })();
            // Update the best_fire with the one with lowest access time
            //self::println!("lelo: {:?}, {:?}", point, point);
            best_fire = match (point, best_fire) {
                (Some(point), Some(best_fire)) if point.time < best_fire.time => Some(point),
                (Some(point), None) => Some(point),
                _ => best_fire,
            };
        }
        ///////////////////////////////////////////////////
        // End of neighbor analysys, save point if improves
        ///////////////////////////////////////////////////
        match (me, best_fire) {
            (Some(me), Some(best_fire)) if best_fire.time < me.time => {
                best_fire.save(global_ix, time, refs_x, refs_y)
            }
            (Some(me), _) => me.save(global_ix, time, refs_x, refs_y),
            (None, Some(best_fire)) => best_fire.save(global_ix, time, refs_x, refs_y),
            _ => (),
        }
    }
}

// TODO: Fine-tune these constants and make them configurable
fn similar_fires(a: FireSimpleCuda, b: FireSimpleCuda) -> bool {
    (a.speed_max - b.speed_max).abs() < 1.0
        && (a.azimuth_max - b.azimuth_max).abs() < 5.0 / (2.0 * core::f32::consts::PI)
        && (a.eccentricity - b.eccentricity).abs() < 0.1
}

fn compute_shared_ix(pos: &USizeVec2) -> Option<usize> {
    let shared_x = pos.x % (BLOCK_WIDTH);
    let shared_y = pos.y % (BLOCK_HEIGHT);
    let shared_bx = pos.x / (BLOCK_WIDTH);
    let shared_by = pos.y / (BLOCK_HEIGHT);
    let my_bx = thread::block_idx_x() as usize;
    let my_by = thread::block_idx_y() as usize;
    let is_top_neighbor = shared_by == my_by - 1;
    let is_bottom_neighbor = shared_by == my_by + 1;
    let is_left_neighbor = shared_bx == my_bx - 1;
    let is_right_neighbor = shared_bx == my_bx + 1;

    if let Some((shared_x, shared_y)) = match (
        is_top_neighbor,
        is_bottom_neighbor,
        is_left_neighbor,
        is_right_neighbor,
    ) {
        //TL
        (true, false, true, false) => Some((0, 0)),
        // T
        (true, false, false, false) => Some((shared_x + 1, 0)),
        // TR
        (true, false, false, true) => Some((BLOCK_WIDTH + 1, 0)),
        // R
        (false, false, false, true) => Some((BLOCK_WIDTH + 1, shared_y + 1)),
        // BR
        (false, true, false, true) => Some((BLOCK_WIDTH + 1, BLOCK_HEIGHT + 1)),
        // B
        (false, true, false, false) => Some((shared_x + 1, 0)),
        // BL
        (false, true, true, false) => Some((0, BLOCK_HEIGHT + 1)),
        // L
        (false, false, true, false) => Some((0, shared_y + 1)),
        // This block
        (false, false, false, false) if shared_bx == my_bx && shared_by == my_by => {
            Some((shared_x + 1, shared_y + 1))
        }

        _ => None,
    } {
        Some((shared_x + (shared_y) * (BLOCK_WIDTH + 2)))
    } else {
        None
    }
}

impl PointRef {
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
        let speed = self.fire.spread(bearing).speed().get::<meter_per_second>();
        let distance = geo_ref.distance(from_pos, to);
        let time = self.time;
        if speed > 1e-6 {
            Some(time + (distance / speed))
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, cust_core::DeviceCopy)]
#[repr(C)]
struct PointRef {
    time: f32,
    pos_x: usize,
    pos_y: usize,
    fire: FireSimpleCuda,
}

impl PointRef {
    fn pos(&self) -> USizeVec2 {
        USizeVec2 {
            x: self.pos_x,
            y: self.pos_y,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, cust_core::DeviceCopy)]
#[repr(C)]
struct Point {
    time: f32,
    fire: Option<FireSimpleCuda>,
    reference: PointRef,
}

impl Point {
    unsafe fn load(
        geo_ref: &GeoReference,
        pos: USizeVec2,
        idx: usize,
        speed_max: &[Option<float::T>],
        azimuth_max: &[Option<float::T>],
        eccentricity: &[Option<float::T>],
        time: *const Option<float::T>,
        ref_x: *const Option<usize>,
        ref_y: *const Option<usize>,
    ) -> Option<Self> {
        if read_volatile(time.add(idx)).is_some() {
            //self::println!("time: {:?}", pos);
        };
        let p_time = read_volatile(time.add(idx))?;
        let fire = load_fire(idx, speed_max, azimuth_max, eccentricity);
        /*
        if time[idx].is_some() {
            self::println!("fogo: {:?}", pos);
        };
        */
        let ref_pos = USizeVec2 {
            x: read_volatile(ref_x.add(idx))?,
            y: read_volatile(ref_y.add(idx))?,
        };
        /*
        if time[idx].is_some() {
            self::println!("ref_pos: {:?}", ref_pos);
        };
        */
        let reference = match (fire) {
            (Some(_)) if ref_pos != pos => {
                let ref_ix: usize = ref_pos.x + ref_pos.y * geo_ref.size[0] as usize;
                /*
                self::println!(
                    "fuego tiemposo! {:?} {:?} {:?}",
                    pos,
                    ref_pos,
                    time[ref_ix].is_some()
                );
                */
                let fire = load_fire(ref_ix, speed_max, azimuth_max, eccentricity)?;
                //self::println!("fuego tiemposo1 {} {:?}", time[ref_ix].is_some(), (ref_ix%1000, ref_ix.div(1000)));
                let r_time = read_volatile(time.add(ref_ix))?;
                //self::println!("fuego tiemposo2");
                Some(PointRef {
                    time: r_time,
                    pos_x: ref_pos.x,
                    pos_y: ref_pos.y,
                    fire,
                })
            }
            (Some(fire)) => Some(PointRef {
                time: p_time,
                pos_x: pos.x,
                pos_y: pos.y,
                fire,
            }),
            _ => None,
        };
        /*
        self::println!(
            "fuego tiempo! {:?} {:?} {:?}",
            pos,
            fire.is_some(),
            reference.is_some()
        );
        */
        Some(Point {
            time: p_time,
            fire,
            reference: reference?,
        })
    }

    unsafe fn save(
        &self,
        idx: usize,
        time: *mut Option<float::T>,
        ref_x: *mut Option<usize>,
        ref_y: *mut Option<usize>,
    ) {
        let pos = (idx % 1000, idx / 1000);
        write_volatile(time.add(idx), Some(self.time));
        write_volatile(ref_x.add(idx), Some(self.reference.pos_x));
        write_volatile(ref_y.add(idx), Some(self.reference.pos_y));
    }
    fn as_reference(&self, pos: USizeVec2) -> Option<PointRef> {
        Some(PointRef {
            time: self.time,
            pos_x: pos.x,
            pos_y: pos.y,
            fire: self.fire?,
        })
    }
}

fn load_fire(
    idx: usize,
    speed_max: &[Option<float::T>],
    azimuth_max: &[Option<float::T>],
    eccentricity: &[Option<float::T>],
) -> Option<FireSimpleCuda> {
    Some(
        (FireSimpleCuda {
            azimuth_max: azimuth_max[idx]?,
            speed_max: speed_max[idx]?,
            eccentricity: eccentricity[idx]?,
        })
        .into(),
    )
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
            // Neighbor position in global pixel coords
            let pos = IVec2 {
                x: idx_2d.x as i32 + di,
                y: idx_2d.y as i32 + dj,
            };
            // If this neighbor is outside this block or global area
            // continue
            if pos.x < 0 || pos.x >= width as _ || pos.y < 0 || pos.y >= height as _ {
                None
            } else {
                let pos = USizeVec2 {
                    x: pos.x as usize,
                    y: pos.y as usize,
                };
                let shared_mem_ix = compute_shared_ix(&pos)?;
                let point = (*shared.add(shared_mem_ix))?;
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
