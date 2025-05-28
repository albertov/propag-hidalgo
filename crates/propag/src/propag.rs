use ::firelib::float;
use ::firelib::float::*;
use ::firelib::*;
use geometry::GeoReference;
use soa_derive::StructOfArray;

//FIXME: Use the C version as source of truth with bindgen
pub const HALO_RADIUS: i32 = 1;

#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct Settings {
    pub geo_ref: GeoReference,
    pub max_time: f32,
    pub find_ref_change: bool,
}

impl Settings {
    #[unsafe(no_mangle)]
    // This one is so cbindgen export FireSimpleCuda
    pub extern "C" fn create_settings(
        geo_ref: GeoReference,
        max_time: f32,
        find_ref_change: bool,
    ) -> Self {
        Self {
            geo_ref,
            max_time,
            find_ref_change,
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
    // This one is so cbindgen export FireSimpleCuda
    #[unsafe(no_mangle)]
    pub extern "C" fn create_fire(
        speed_max: float::T,
        azimuth_max: float::T,
        eccentricity: float::T,
    ) -> FireSimpleCuda {
        Self {
            speed_max,
            azimuth_max,
            eccentricity,
        }
    }
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
