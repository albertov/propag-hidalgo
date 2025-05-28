use ::firelib::float;
use ::firelib::float::*;
use ::firelib::*;
use core::ffi::c_char;
use gdal::errors::GdalError::CplError;
use gdal::raster::Buffer;
use gdal::raster::RasterCreationOptions;
use gdal::spatial_ref::SpatialRef;
use gdal::DriverManager;
use gdal_sys::CPLErr::CE_None;
use gdal_sys::OGRGeometryH;
use geometry::GeoReference;
use min_max_traits::Max;
use soa_derive::StructOfArray;
use std::ffi::CStr;

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
    pub settings: Settings,
    pub output_path: String,
    pub initial_ignited_elements: Vec<TimeFeature>,
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
struct FFITimeFeature {
    time: float::T,
    geom_wkb: *const u8,
    geom_wkb_len: usize,
}
#[cfg(not(target_os = "cuda"))]
impl TryFrom<&FFITimeFeature> for TimeFeature {
    type Error = gdal::errors::GdalError;
    fn try_from(f: &FFITimeFeature) -> Result<Self, Self::Error> {
        let s = unsafe { std::slice::from_raw_parts(f.geom_wkb, f.geom_wkb_len) };
        let geom = gdal::vector::Geometry::from_wkb(s)?;
        Ok(TimeFeature { time: f.time, geom })
    }
}

#[repr(C)]
pub struct FFIPropagation {
    settings: Settings,
    output_path: *const c_char,
    initial_ignited_elements: *const FFITimeFeature,
    initial_ignited_elements_len: usize,
    terrain_loader: FFITerrainLoader,
}

impl From<FFIPropagation> for Propagation {
    fn from(p: FFIPropagation) -> Self {
        let is = unsafe {
            std::slice::from_raw_parts(p.initial_ignited_elements, p.initial_ignited_elements_len)
        };
        let initial_ignited_elements = is
            .iter()
            .filter_map(|x| TimeFeature::try_from(x).ok())
            .collect();
        let output_path = unsafe { CStr::from_ptr(p.output_path) };
        let output_path = String::from_utf8_lossy(output_path.to_bytes()).to_string();
        Propagation {
            settings: p.settings,
            output_path,
            initial_ignited_elements,
            terrain_loader: Box::new(p.terrain_loader),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn FFIPropagation_run(propag: FFIPropagation) {
    let propag: Propagation = propag.into();
    println!("settings={:?}", propag.settings);
    let geo_ref = propag.settings.geo_ref;
    if let Ok(times) = rasterize_times(&propag.initial_ignited_elements, &geo_ref) {
        let _ = write_times(times, &geo_ref, propag.output_path);
    }
}

fn write_times(
    times: Vec<f32>,
    geo_ref: &GeoReference,
    output_path: String,
) -> gdal::errors::Result<()> {
    let gtiff = DriverManager::get_driver_by_name("GTIFF")?;
    let options =
        RasterCreationOptions::from_iter(["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]);
    let mut ds = gtiff.create_with_band_type_with_options::<f32, _>(
        output_path,
        geo_ref.width as usize,
        geo_ref.height as usize,
        1,
        &options,
    )?;
    let srs = SpatialRef::from_epsg(geo_ref.epsg)?;
    ds.set_spatial_ref(&srs)?;
    ds.set_geo_transform(&geo_ref.transform.as_array_64())?;
    let mut band = ds.rasterband(1)?;
    let no_data: f32 = Max::MAX;
    band.set_no_data_value(Some(no_data as f64))?;
    let mut buf = Buffer::new(band.size(), times);
    band.write((0, 0), band.size(), &mut buf)?;
    Ok(())
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

pub fn rasterize_times(
    times: &[TimeFeature],
    geo_ref: &GeoReference,
) -> gdal::errors::Result<Vec<f32>> {
    let d = DriverManager::get_driver_by_name("MEM")?;
    let mut ds = d.create("in-memory", geo_ref.width as _, geo_ref.height as _, 1)?;
    ds.set_spatial_ref(&SpatialRef::from_epsg(geo_ref.epsg)?)?;
    ds.set_geo_transform(&geo_ref.transform.as_array_64())?;
    let mut b = ds.rasterband(1)?;
    let no_data: f32 = Max::MAX;
    b.set_no_data_value(Some(no_data as f64))?;
    b.fill(no_data as f64, None)?;
    let time_values: Vec<f64> = times.iter().map(|f| f.time as f64).collect();
    let ret = unsafe {
        let geoms: Vec<OGRGeometryH> = times.iter().map(|f| f.geom.c_geometry()).collect();
        gdal_sys::GDALRasterizeGeometries(
            ds.c_dataset(),
            1,
            [1i32].as_ptr(),
            geoms.len() as _,
            geoms.as_ptr(),
            None,
            core::ptr::null_mut(),
            time_values.as_ptr(),
            core::ptr::null_mut(),
            None,
            core::ptr::null_mut(),
        )
    };
    if ret == CE_None {
        ds.flush_cache()?;
        let b = ds.rasterband(1)?;
        let buf = b.read_band_as()?;
        Ok(buf.into_shape_and_vec().1)
    } else {
        Err(CplError {
            class: ret,
            number: -1,
            msg: "GDALRasterizeGeometries".to_string(),
        })
    }
}
