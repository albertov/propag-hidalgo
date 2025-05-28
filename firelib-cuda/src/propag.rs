use core::ops::Div;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use cuda_std::shared::dynamic_shared_mem;
use cuda_std::thread::*;
use cuda_std::*;
use firelib_rs::float;
use firelib_rs::float::*;
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

//FIXME: Use the C version as source of truth with bindgen
pub const HALO_RADIUS: usize = 1;
#[cfg(not(target_os = "cuda"))]
pub const PI: f32 = 3.141592653589793;

//FIXME: Use the C version as source of truth with bindgen
pub const MAX_TIME: f32 = 340282346638528859811704183484516925440.0;

pub const SIZEOF_FBC_SHARED_ITEM: usize = core::mem::size_of::<Point>();

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_boundary_changes(
    width: u32,
    height: u32,
    refs_x: *const u16,
    refs_y: *const u16,
    out: *mut bool,
) {
    let shared: *mut Point = dynamic_shared_mem();
    let global = IVec2 {
        x: thread::index_2d().x as i32,
        y: thread::index_2d().y as i32,
    };
    let in_bounds = global.x < width as i32 && global.y < height as i32;

    let local = IVec2 {
        x: thread::thread_idx_x() as i32 + HALO_RADIUS as i32,
        y: thread::thread_idx_y() as i32 + HALO_RADIUS as i32,
    };
    let shared_width = thread::block_dim_x() as i32 + HALO_RADIUS as i32 * 2;

    let load_point_at_offset = |ox: i32, oy: i32| {
        let local = IVec2 {
            x: local.x + ox,
            y: local.y + oy,
        };
        let global = IVec2 {
            x: global.x + ox,
            y: global.y + oy,
        };
        let local_ix = local.x + local.y * shared_width;
        core::ptr::write_volatile(shared.add(local_ix as usize), {
            if global.x >= 0 && global.x < width as i32 && global.y >= 0 && global.y < height as i32
            {
                let global_ix = global.x + global.y * width as i32;
                Point {
                    x: *refs_x.add(global_ix as usize),
                    y: *refs_y.add(global_ix as usize),
                }
            } else {
                Point {
                    x: Max::MAX,
                    y: Max::MAX,
                }
            }
        });
    };

    if in_bounds {
        let x_near_x0 = thread::thread_idx_x() < HALO_RADIUS as u32;
        let y_near_y0 = thread::thread_idx_y() < HALO_RADIUS as u32;

        if y_near_y0 {
            load_point_at_offset(0, -(HALO_RADIUS as i32));
            load_point_at_offset(0, (thread::block_dim_x() as i32));
        }
        if x_near_x0 {
            load_point_at_offset(-(HALO_RADIUS as i32), 0);
            load_point_at_offset((thread::block_dim_x() as i32), 0);
        }
        if x_near_x0 && y_near_y0 {
            load_point_at_offset(-(HALO_RADIUS as i32), -(HALO_RADIUS as i32));
            load_point_at_offset((thread::block_dim_x() as i32), -(HALO_RADIUS as i32));
            load_point_at_offset(
                (thread::block_dim_x() as i32),
                (thread::block_dim_y() as i32),
            );
            load_point_at_offset(-(HALO_RADIUS as i32), (thread::block_dim_y() as i32));
        }
    }

    thread::sync_threads();

    if in_bounds {
        let local_ix = (local.x + local.y * shared_width) as usize;
        let me = core::ptr::read_volatile(shared.add(local_ix));
        let mut result = false;
        for i in (-1..2) {
            for j in (-1..2) {
                let p = core::ptr::read_volatile(
                    shared.add(((local.x + i) + (local.y + j) * shared_width) as usize),
                );
                result |= p.x != Max::MAX && p.y != Max::MAX && (p.x != me.x || p.y != me.y);
            }
        }
        let global_ix = (global.x + global.y * width as i32) as usize;
        if result {
            //self::println!("global_ix={}, result={}", global_ix, result);
        }
        *out.add(global_ix) = result;
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C, align(16))]
struct Point {
    x: u16,
    y: u16,
}

//FIXME: Use the C version as source of truth with bindgen
#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct Settings {
    pub geo_ref: GeoReference,
    pub max_time: f32,
}

#[cfg_attr(not(target_os = "cuda"), derive(StructOfArray), soa_derive(Debug))]
#[derive(Copy, Clone, Debug, PartialEq, cust_core::DeviceCopy)]
#[repr(C, align(16))]
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
