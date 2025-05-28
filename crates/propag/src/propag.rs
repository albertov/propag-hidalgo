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

pub struct Propagation {
    pub max_time: f32,
    pub find_ref_change: bool,
    pub epsg: u32,
    pub res_x: f32,
    pub res_y: f32,
    pub ignited_elements: Vec<TimeFeature>,
    pub terrain_loader: Box<dyn TerrainLoader>,
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

pub struct TimeFeature {
    time: float::T,
    geom: gdal::vector::Geometry,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct FFITimeFeature {
    time: float::T,
    geom_wkb: *const u8,
    geom_wkb_len: usize,
}
#[cfg(not(target_os = "cuda"))]
impl TryFrom<FFITimeFeature> for TimeFeature {
    type Error = gdal::errors::GdalError;
    fn try_from(f: FFITimeFeature) -> Result<Self, Self::Error> {
        let s = unsafe { std::slice::from_raw_parts(f.geom_wkb, f.geom_wkb_len) };
        let geom = gdal::vector::Geometry::from_wkb(s)?;
        Ok(TimeFeature { time: f.time, geom })
    }
}

#[repr(C)]
pub struct FFIPropagation {
    max_time: f32,
    find_ref_change: bool,
    epsg: u32,
    pub res_x: f32,
    pub res_y: f32,
    ignited_elements: *const FFITimeFeature,
    ignited_elements_len: usize,
    terrain_loader: FFITerrainLoader,
}

impl From<FFIPropagation> for Propagation {
    fn from(p: FFIPropagation) -> Self {
        let is = unsafe { std::slice::from_raw_parts(p.ignited_elements, p.ignited_elements_len) };
        let ignited_elements = is
            .iter()
            .filter_map(|x| TimeFeature::try_from(*x).ok())
            .collect();
        Propagation {
            max_time: p.max_time,
            find_ref_change: p.find_ref_change,
            epsg: p.epsg,
            res_x: p.res_x,
            res_y: p.res_y,
            ignited_elements,
            terrain_loader: Box::new(p.terrain_loader),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ffi_run_propagation(propag: &FFIPropagation) {
    todo!()
}

pub trait TerrainLoader {
    fn load_extent(&self, geo_ref: &geometry::GeoReference) -> Option<TerrainCudaVec>;
}

#[repr(C)]
struct FFITerrain {
    fuel_code: *mut u8,
    d1hr: *mut float::T,
    d10hr: *mut float::T,
    d100hr: *mut float::T,
    herb: *mut float::T,
    wood: *mut float::T,
    wind_speed: *mut float::T,
    wind_azimuth: *mut float::T,
    slope: *mut float::T,
    aspect: *mut float::T,
}

#[repr(C)]
struct FFITerrainLoader {
    load: LoadFn,
    context: *mut core::ffi::c_void,
}

type LoadFn = unsafe extern "C" fn(*mut core::ffi::c_void, &GeoReference, &mut FFITerrain) -> bool;

impl TerrainLoader for FFITerrainLoader {
    fn load_extent(&self, geo_ref: &geometry::GeoReference) -> Option<TerrainCudaVec> {
        let mut ret = TerrainCudaVec::with_capacity(geo_ref.len() as _);
        let mut chunk = FFITerrain {
            fuel_code: ret.fuel_code.as_mut_ptr(),
            d1hr: ret.d1hr.as_mut_ptr(),
            d10hr: ret.d10hr.as_mut_ptr(),
            d100hr: ret.d100hr.as_mut_ptr(),
            herb: ret.herb.as_mut_ptr(),
            wood: ret.wood.as_mut_ptr(),
            wind_speed: ret.wind_speed.as_mut_ptr(),
            wind_azimuth: ret.wind_azimuth.as_mut_ptr(),
            slope: ret.slope.as_mut_ptr(),
            aspect: ret.aspect.as_mut_ptr(),
        };
        if unsafe { (self.load)(self.context, geo_ref, &mut chunk) } {
            Some(ret)
        } else {
            None
        }
    }
}
