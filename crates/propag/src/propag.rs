use ::geometry::*;
use gdal::Dataset;

use core::ffi::c_char;
use cust::device::DeviceAttribute;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::prelude::*;
use firelib::float::*;
use firelib::TerrainCuda;
use firelib::{float, from_quantity, to_quantity, FireSimple, TerrainCudaVec};
use gdal::errors::GdalError;
use gdal::errors::GdalError::CplError;
use gdal::raster::Buffer;
use gdal::raster::RasterCreationOptions;
use gdal::spatial_ref::SpatialRef;
use gdal::vector::Geometry;
use gdal::vector::LayerAccess;
use gdal::vector::LayerOptions;
use gdal::DriverManager;
use gdal_sys::CPLErr::CE_None;
use gdal_sys::OGRGeometryH;
use geometry::GeoReference;
use min_max_traits::Max;
use soa_derive::StructOfArray;
use std::ffi::CStr;
use std::fmt;

//FIXME: Use the C version as source of truth with bindgen
pub const HALO_RADIUS: i32 = 3;

const THREAD_BLOCK_AXIS_LENGTH: u32 = 19;

static PTX: &str = include_str!("../../target/cuda/firelib.ptx");
static PTX_C: &str = include_str!("../../target/cuda/propag_c.ptx");

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

#[cfg_attr(not(target_os = "cuda"), derive(Debug))]
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
        if !f.geom_wkb.is_null() && f.geom_wkb_len > 0 {
            let s = unsafe { std::slice::from_raw_parts(f.geom_wkb, f.geom_wkb_len) };
            let geom = gdal::vector::Geometry::from_wkb(s)?;
            Ok(TimeFeature { time: f.time, geom })
        } else {
            Err(CplError {
                class: 1,
                number: -1,
                msg: "invalid wkb geom".to_string(),
            })
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(Debug))]
pub struct FuelFeature {
    code: u8,
    geom: gdal::vector::Geometry,
}

#[repr(C)]
pub struct FFIFuelFeature {
    code: u8,
    geom_wkb: *const u8,
    geom_wkb_len: usize,
}
#[cfg(not(target_os = "cuda"))]
impl TryFrom<&FFIFuelFeature> for FuelFeature {
    type Error = gdal::errors::GdalError;
    fn try_from(f: &FFIFuelFeature) -> Result<Self, Self::Error> {
        let s = unsafe { std::slice::from_raw_parts(f.geom_wkb, f.geom_wkb_len) };
        let geom = gdal::vector::Geometry::from_wkb(s)?;
        Ok(FuelFeature { code: f.code, geom })
    }
}

pub struct Propagation {
    pub settings: Settings,
    pub output_path: String,
    pub refs_output_path: Option<String>,
    pub block_boundaries_out_path: Option<String>,
    pub grid_boundaries_out_path: Option<String>,
    pub initial_ignited_elements: Vec<TimeFeature>,
    pub initial_ignited_elements_crs: SpatialRef,
    pub terrain_loader: Box<dyn TerrainLoader>,
}

#[repr(C)]
pub struct FFIPropagation {
    settings: Settings,
    output_path: *const c_char,
    refs_output_path: *const c_char,
    block_boundaries_out_path: *const c_char,
    grid_boundaries_out_path: *const c_char,
    initial_ignited_elements: *const FFITimeFeature,
    initial_ignited_elements_len: usize,
    initial_ignited_elements_crs: *const c_char,
    terrain_loader: FFITerrainLoader,
}

fn some_path(p: *const c_char) -> Option<String> {
    if !p.is_null() {
        let out = unsafe { CStr::from_ptr(p) };
        match String::from_utf8_lossy(out.to_bytes()).to_string() {
            x if x.is_empty() => None,
            x => Some(x),
        }
    } else {
        None
    }
}

impl TryFrom<FFIPropagation> for Propagation {
    type Error = gdal::errors::GdalError;
    fn try_from(p: FFIPropagation) -> Result<Self, Self::Error> {
        let initial_ignited_elements =
            if !p.initial_ignited_elements.is_null() && p.initial_ignited_elements_len > 0 {
                let is = unsafe {
                    std::slice::from_raw_parts(
                        p.initial_ignited_elements,
                        p.initial_ignited_elements_len,
                    )
                };
                is.iter()
                    .filter_map(|x| TimeFeature::try_from(x).ok())
                    .collect()
            } else {
                Vec::new()
            };
        let initial_ignited_elements_crs =
            unsafe { spatial_ref_from_buf(p.initial_ignited_elements_crs) }?;
        let output_path = some_path(p.output_path).ok_or(CplError {
            class: 1,
            number: -1,
            msg: "invalid output_path".to_string(),
        })?;
        let refs_output_path = some_path(p.refs_output_path);
        let block_boundaries_out_path = some_path(p.block_boundaries_out_path);
        let grid_boundaries_out_path = some_path(p.grid_boundaries_out_path);
        Ok(Propagation {
            settings: p.settings,
            output_path,
            block_boundaries_out_path,
            grid_boundaries_out_path,
            refs_output_path,
            initial_ignited_elements,
            initial_ignited_elements_crs,
            terrain_loader: Box::new(p.terrain_loader),
        })
    }
}

unsafe fn spatial_ref_from_buf(buf: *const c_char) -> gdal::errors::Result<SpatialRef> {
    let crs = unsafe { CStr::from_ptr(buf) };
    let crs = String::from_utf8_lossy(crs.to_bytes()).to_string();
    SpatialRef::from_proj4(&crs)
}

#[derive(Debug, Clone)]
pub enum PropagError {
    PropagGdalError(GdalError),
    PropagCudaError(CudaError),
    PropagReadPTXError(String),
    PropagLoadExtentError,
}

impl fmt::Display for PropagError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::PropagError::*;
        match self {
            PropagGdalError(err) => write!(f, "{}", err),
            PropagCudaError(err) => write!(f, "{}", err),
            PropagReadPTXError(err) => write!(f, "{}", err),
            PropagLoadExtentError => write!(f, "load_extent_error"),
        }
    }
}

impl std::error::Error for PropagError {}

#[unsafe(no_mangle)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn FFIPropagation_run(
    propag: FFIPropagation,
    err_len: usize,
    err_msg: *mut c_char,
) -> bool {
    use self::PropagError::*;
    match std::panic::catch_unwind(|| {
        let propag: Propagation = propag.try_into().map_err(PropagGdalError)?;
        let geo_ref = propag.settings.geo_ref;
        let time = rasterize_times(
            &propag.initial_ignited_elements,
            propag.initial_ignited_elements_crs.clone(),
            &geo_ref,
        )
        .map_err(PropagGdalError)?;
        let propag_result = propagate(&propag, time)?;
        println!(
            "block_size={:?}, grid_size={:?}, super_grid_size={:?}",
            &propag_result.block_size, &propag_result.grid_size, &propag_result.super_grid_size
        );
        write_times(&propag_result.time, &geo_ref, &propag.output_path).map_err(PropagGdalError)?;
        if let Some(ref out_path) = propag.refs_output_path {
            write_refs(&propag_result, out_path).map_err(PropagGdalError)?;
        }
        write_boundaries(&propag_result, &propag).map_err(PropagGdalError)?;
        Ok::<(), PropagError>(())
    }) {
        Ok(Ok(())) => true,
        Ok(Err(err)) => {
            let err = format!("{}", err);
            if let Ok(c_err) = std::ffi::CString::new(err) {
                libc::strncpy(err_msg, c_err.as_ptr(), err_len);
            }
            false
        }
        Err(err) => {
            let err = format!("{:?}", err);
            if let Ok(c_err) = std::ffi::CString::new(err) {
                libc::strncpy(err_msg, c_err.as_ptr(), err_len);
            }
            false
        }
    }
}

#[derive(Debug)]
pub struct PropagResults {
    pub geo_ref: GeoReference,
    pub time: Vec<f32>,
    pub boundary_change: Vec<u16>,
    pub refs_x: Vec<u16>,
    pub refs_y: Vec<u16>,
    pub refs_time: Vec<f32>,
    pub grid_size: GridSize,
    pub block_size: BlockSize,
    pub super_grid_size: GridSize,
}

fn propagate(propag: &Propagation, mut time: Vec<f32>) -> Result<PropagResults, PropagError> {
    let settings = propag.settings;
    let geo_ref = settings.geo_ref;
    let len = geo_ref.len() as usize;
    use PropagError::*;
    let terrain = propag
        .terrain_loader
        .load_extent(&geo_ref)
        .map(Ok)
        .unwrap_or(Err(PropagLoadExtentError))?;
    cust::init(CudaFlags::empty()).map_err(PropagCudaError)?;
    let device = Device::get_device(0).map_err(PropagCudaError)?;
    let ctx = Context::new(device).map_err(PropagCudaError)?;
    ctx.set_flags(ContextFlags::SCHED_AUTO)
        .map_err(PropagCudaError)?;

    let module_c = if let Ok(path) = std::env::var("PROPAG_PTX") {
        let contents =
            std::fs::read_to_string(path).map_err(|x| PropagReadPTXError(format!("{}", x)))?;
        Module::from_ptx(contents, &[]).map_err(PropagCudaError)?
    } else {
        Module::from_ptx(PTX_C, &[]).map_err(PropagCudaError)?
    };

    (|| {
        // Make the CUDA module, modules just house the GPU code for the kernels we created.
        // they can be made from PTX code, cubins, or fatbins.
        let module = Module::from_ptx(PTX, &[])?;

        // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // retrieve the add kernel from the module so we can calculate the right launch config.
        let propag_c = module_c.get_function("propag")?;
        let fixup = module_c.get_function("fixup")?;
        let pre_burn = module.get_function("cuda_standard_simple_burn")?;

        let block_size = BlockSize {
            x: THREAD_BLOCK_AXIS_LENGTH,
            y: THREAD_BLOCK_AXIS_LENGTH,
            z: 1,
        };

        let radius = HALO_RADIUS as u32;
        let shmem_size = (block_size.x + radius * 2) * (block_size.y + radius * 2);
        let shmem_bytes = shmem_size * 24; //FIXME: std::mem::size_of::<Point>() as u32;
                                           //assert_eq!(std::mem::size_of::<Point>(), 64);

        let max_active_blocks: u32 =
            propag_c.max_active_blocks_per_multiprocessor(block_size, shmem_bytes as _)?;
        let mp_count: i32 = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
        let max_total_blocks: u32 = mp_count as u32 * max_active_blocks;

        // Assumes square block size
        assert_eq!(block_size.x, block_size.y);
        let grid_w = (max_total_blocks as f64).sqrt().floor() as u32;
        let grid_size: GridSize = (grid_w, grid_w).into();

        let super_grid_size: GridSize = (
            geo_ref.width.div_ceil(grid_size.x * block_size.x),
            geo_ref.height.div_ceil(grid_size.y * block_size.y),
        )
            .into();

        // allocate the GPU memory needed to house our numbers and copy them over.
        let model_gpu = terrain.fuel_code.as_slice().as_dbuf()?;
        let d1hr_gpu = terrain.d1hr.as_slice().as_dbuf()?;
        let d10hr_gpu = terrain.d10hr.as_slice().as_dbuf()?;
        let d100hr_gpu = terrain.d100hr.as_slice().as_dbuf()?;
        let herb_gpu = terrain.herb.as_slice().as_dbuf()?;
        let wood_gpu = terrain.wood.as_slice().as_dbuf()?;
        let wind_speed_gpu = terrain.wind_speed.as_slice().as_dbuf()?;
        let wind_azimuth_gpu = terrain.wind_azimuth.as_slice().as_dbuf()?;
        let slope_gpu = terrain.slope.as_slice().as_dbuf()?;
        let aspect_gpu = terrain.aspect.as_slice().as_dbuf()?;

        // input/output vectors
        let mut refs_x: Vec<u16> = std::iter::repeat_n(Max::MAX, len).collect();
        let mut refs_y: Vec<u16> = std::iter::repeat_n(Max::MAX, len).collect();
        let mut refs_time: Vec<f32> = std::iter::repeat_n(Max::MAX, len).collect();
        let mut boundary_change: Vec<u16> = std::iter::repeat_n(0, len).collect();

        let speed_max: Vec<float::T> = std::iter::repeat_n(0.0, len).collect();
        let azimuth_max: Vec<float::T> = std::iter::repeat_n(0.0, len).collect();
        let eccentricity: Vec<float::T> = std::iter::repeat_n(0.0, len).collect();

        for j in 0..geo_ref.height {
            for i in 0..geo_ref.width {
                let ix = (i + j * geo_ref.width) as usize;
                if time[ix] != Max::MAX {
                    refs_x[ix] = i as u16;
                    refs_y[ix] = j as u16;
                    refs_time[ix] = time[ix];
                }
            }
        }

        let speed_max_buf = speed_max.as_slice().as_dbuf()?;
        let azimuth_max_buf = azimuth_max.as_slice().as_dbuf()?;
        let eccentricity_buf = eccentricity.as_slice().as_dbuf()?;
        let refs_x_buf = refs_x.as_slice().as_dbuf()?;
        let refs_y_buf = refs_y.as_slice().as_dbuf()?;
        let refs_time_buf = refs_time.as_slice().as_dbuf()?;
        let time_buf = time.as_slice().as_dbuf()?;
        let boundary_change_buf = boundary_change.as_slice().as_dbuf()?;
        let (_, pre_burn_block_size) = pre_burn.suggested_launch_configuration(0, 0.into())?;
        let pre_burn_grid_size = geo_ref.len().div_ceil(pre_burn_block_size);
        unsafe {
            launch!(
                pre_burn<<<pre_burn_grid_size, pre_burn_block_size, 0, stream>>>(
                    len,
                    model_gpu.as_device_ptr(),
                    d1hr_gpu.as_device_ptr(),
                    d10hr_gpu.as_device_ptr(),
                    d100hr_gpu.as_device_ptr(),
                    herb_gpu.as_device_ptr(),
                    wood_gpu.as_device_ptr(),
                    wind_speed_gpu.as_device_ptr(),
                    wind_azimuth_gpu.as_device_ptr(),
                    slope_gpu.as_device_ptr(),
                    aspect_gpu.as_device_ptr(),
                    speed_max_buf.as_device_ptr(),
                    azimuth_max_buf.as_device_ptr(),
                    eccentricity_buf.as_device_ptr(),
                )
            )?;
            stream.synchronize()?;
            loop {
                let mut worked: Vec<DeviceVariable<u32>> =
                    Vec::with_capacity((super_grid_size.x * super_grid_size.y) as usize);
                for grid_x in 0..super_grid_size.x {
                    for grid_y in 0..super_grid_size.y {
                        let this_worked = DeviceVariable::new(0)?;
                        let progress = DeviceVariable::new(0)?;
                        cust::launch_cooperative!(
                            // slices are passed as two parameters, the pointer and the length.
                            propag_c<<<grid_size, block_size, shmem_bytes, stream>>>(
                                settings,
                                grid_x,
                                grid_y,
                                this_worked.as_device_ptr(),
                                speed_max_buf.as_device_ptr(),
                                azimuth_max_buf.as_device_ptr(),
                                eccentricity_buf.as_device_ptr(),
                                time_buf.as_device_ptr(),
                                refs_x_buf.as_device_ptr(),
                                refs_y_buf.as_device_ptr(),
                                refs_time_buf.as_device_ptr(),
                                boundary_change_buf.as_device_ptr(),
                                progress.as_device_ptr(),
                            )
                        )?;
                        worked.push(this_worked);
                    }
                }
                stream.synchronize()?;
                if worked.iter_mut().all(|x| {
                    let _ = x.copy_dtoh();
                    **x == 0
                }) {
                    break;
                }
            }
            stream.synchronize()?;
            for grid_x in 0..super_grid_size.x {
                for grid_y in 0..super_grid_size.y {
                    cust::launch!(
                        fixup<<<grid_size, block_size, shmem_bytes, stream>>>(
                            settings,
                            grid_x,
                            grid_y,
                            speed_max_buf.as_device_ptr(),
                            azimuth_max_buf.as_device_ptr(),
                            eccentricity_buf.as_device_ptr(),
                            time_buf.as_device_ptr(),
                            refs_x_buf.as_device_ptr(),
                            refs_y_buf.as_device_ptr(),
                            refs_time_buf.as_device_ptr(),
                            boundary_change_buf.as_device_ptr(),
                        )
                    )?;
                }
            }
        };
        time_buf.copy_to(&mut time)?;
        if propag.settings.find_ref_change {
            boundary_change_buf.copy_to(&mut boundary_change)?;
            refs_x_buf.copy_to(&mut refs_x)?;
            refs_y_buf.copy_to(&mut refs_y)?;
            refs_time_buf.copy_to(&mut refs_time)?;
        };
        Ok(PropagResults {
            time,
            boundary_change,
            refs_x,
            refs_y,
            refs_time,
            geo_ref,
            block_size,
            grid_size,
            super_grid_size,
        })
    })()
    .map_err(PropagError::PropagCudaError)
}

fn write_times(
    times: &[f32],
    geo_ref: &GeoReference,
    output_path: &str,
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
    let srs = crate::loader::to_spatial_ref(&geo_ref.proj)?;
    ds.set_spatial_ref(&srs)?;
    ds.set_geo_transform(&geo_ref.transform.as_array_64())?;
    let mut band = ds.rasterband(1)?;
    let no_data: f32 = Max::MAX;
    band.set_no_data_value(Some(no_data as f64))?;
    let mut buf = Buffer::new(band.size(), times.to_vec());
    band.write((0, 0), band.size(), &mut buf)?;
    Ok(())
}
fn write_boundaries(
    PropagResults {
        super_grid_size,
        block_size,
        grid_size,
        geo_ref,
        ..
    }: &PropagResults,
    propag: &Propagation,
) -> gdal::errors::Result<()> {
    let gpkg = DriverManager::get_driver_by_name("GPKG")?;

    let mut grids: Option<Dataset> = if let Some(ref output_path) = &propag.grid_boundaries_out_path
    {
        let mut ds = gpkg.create_vector_only(output_path)?;
        let srs = crate::loader::to_spatial_ref(&geo_ref.proj)?;
        ds.create_layer(LayerOptions {
            name: "grids",
            srs: Some(&srs),
            ty: gdal_sys::OGRwkbGeometryType::wkbPolygon,
            ..Default::default()
        })?;
        Some(ds)
    } else {
        None
    };
    let mut blocks: Option<Dataset> =
        if let Some(ref output_path) = &propag.block_boundaries_out_path {
            let mut ds = gpkg.create_vector_only(output_path)?;
            let srs = crate::loader::to_spatial_ref(&geo_ref.proj)?;
            ds.create_layer(LayerOptions {
                name: "blocks",
                srs: Some(&srs),
                ty: gdal_sys::OGRwkbGeometryType::wkbPolygon,
                ..Default::default()
            })?;
            Some(ds)
        } else {
            None
        };

    let grid_cell_size: UVec2 = (block_size.x * grid_size.x, block_size.y * grid_size.y).into();
    for grid_y in 0..super_grid_size.y {
        for grid_x in 0..super_grid_size.x {
            if let Some(ref mut grids) = grids {
                let a: UVec2 = (grid_x * grid_cell_size.x, grid_y * grid_cell_size.y).into();
                let a = geo_ref.backward(a.as_usizevec2());
                let b: UVec2 = ((grid_x + 1) * grid_cell_size.x, grid_y * grid_cell_size.y).into();
                let b = geo_ref.backward(b.as_usizevec2());
                let c: UVec2 = (
                    (grid_x + 1) * grid_cell_size.x,
                    (grid_y + 1) * grid_cell_size.y,
                )
                    .into();
                let c = geo_ref.backward(c.as_usizevec2());
                let d: UVec2 = (grid_x * grid_cell_size.x, (grid_y + 1) * grid_cell_size.y).into();
                let d = geo_ref.backward(d.as_usizevec2());
                let geom = Geometry::from_wkt(
                    (format!(
                        "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                        a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, a.x, a.y
                    ))
                    .as_str(),
                )?;
                grids.layer(0)?.create_feature(geom)?;
            }
            if let Some(ref mut blocks) = blocks {
                let corner: UVec2 = (grid_x * grid_cell_size.x, grid_y * grid_cell_size.y).into();
                for block_y in 0..grid_size.y {
                    for block_x in 0..grid_size.x {
                        let block: UVec2 = (block_x, block_y).into();
                        let a: UVec2 = (block.x * block_size.x, block.y * block_size.y).into();
                        let a = geo_ref.backward((corner + a).as_usizevec2());
                        let b: UVec2 =
                            ((block.x + 1) * block_size.x, block.y * block_size.y).into();
                        let b = geo_ref.backward((corner + b).as_usizevec2());
                        let c: UVec2 =
                            ((block.x + 1) * block_size.x, (block.y + 1) * block_size.y).into();
                        let c = geo_ref.backward((corner + c).as_usizevec2());
                        let d: UVec2 =
                            (block.x * block_size.x, (block.y + 1) * block_size.y).into();
                        let d = geo_ref.backward((corner + d).as_usizevec2());
                        let geom = Geometry::from_wkt(
                            (format!(
                                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                                a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, a.x, a.y
                            ))
                            .as_str(),
                        )?;
                        blocks.layer(0)?.create_feature(geom)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_refs(
    PropagResults {
        boundary_change,
        refs_x,
        refs_y,
        geo_ref,
        ..
    }: &PropagResults,
    output_path: &str,
) -> gdal::errors::Result<()> {
    let gpkg = DriverManager::get_driver_by_name("GPKG")?;
    let mut ds = gpkg.create_vector_only(output_path)?;
    let srs = crate::loader::to_spatial_ref(&geo_ref.proj)?;
    let mut layer = ds.create_layer(LayerOptions {
        name: "fire_references",
        srs: Some(&srs),
        ty: gdal_sys::OGRwkbGeometryType::wkbLineString,
        ..Default::default()
    })?;
    for i in 0..geo_ref.width {
        for j in 0..geo_ref.height {
            let ix = (i + j * geo_ref.width) as usize;
            if boundary_change[ix] == 1 {
                let dst = geo_ref.backward(USizeVec2 {
                    x: i as usize,
                    y: j as usize,
                });
                let dst = dst
                    + Vec2 {
                        x: geo_ref.transform.dx() / 2.0,
                        y: geo_ref.transform.dy() / 2.0,
                    };
                let src = geo_ref.backward(USizeVec2 {
                    x: refs_x[ix] as usize,
                    y: refs_y[ix] as usize,
                });
                let src = src
                    + Vec2 {
                        x: geo_ref.transform.dx() / 2.0,
                        y: geo_ref.transform.dy() / 2.0,
                    };
                let geom = Geometry::from_wkt(
                    (format!("LINESTRING({} {}, {} {})", src.x, src.y, dst.x, dst.y)).as_str(),
                )?;
                layer.create_feature(geom)?;
            }
        }
    }
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
pub type RunFn = unsafe extern "C" fn(FFIPropagation, usize, *mut c_char) -> bool;

pub type RasterizeFuelsFn = unsafe extern "C" fn(
    *const FFIFuelFeature,
    usize,
    *const c_char,
    &GeoReference,
    *mut u8,
    *mut c_char,
    usize,
) -> bool;

/// This is so the type aliases are exported
#[repr(C)]
pub struct PluginT {
    run: RunFn,
    rasterize_fuels: RasterizeFuelsFn,
}
impl PluginT {
    #[unsafe(no_mangle)]
    // This one is so cbindgen exports PluginT and to ensure RasterizeFuelsFn and RunFn are correct
    pub extern "C" fn create_plugin() -> Self {
        Self {
            run: FFIPropagation_run,
            rasterize_fuels: propag_rasterize_fuels,
        }
    }
}

impl TerrainLoader for FFITerrainLoader {
    fn load_extent(&self, geo_ref: &geometry::GeoReference) -> Option<TerrainCudaVec> {
        let mut ret: TerrainCudaVec =
            std::iter::repeat_n(TerrainCuda::NULL, geo_ref.len() as _).collect();
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
    times_crs: SpatialRef,
    geo_ref: &GeoReference,
) -> gdal::errors::Result<Vec<f32>> {
    let d = DriverManager::get_driver_by_name("MEM")?;
    let mut ds =
        d.create_with_band_type::<f32, _>("in-memory", geo_ref.width as _, geo_ref.height as _, 1)?;
    let ds_srs = crate::loader::to_spatial_ref(&geo_ref.proj)?;
    ds.set_spatial_ref(&ds_srs)?;
    ds.set_geo_transform(&geo_ref.transform.as_array_64())?;
    let mut b = ds.rasterband(1)?;
    let no_data: f32 = Max::MAX;
    b.fill(no_data as f64, None)?;
    b.set_no_data_value(Some(no_data as f64))?;
    let time_values: Vec<f64> = times.iter().map(|f| f.time as f64).collect();
    let geoms: Vec<Geometry> = times
        .iter()
        .filter_map(|f| {
            let mut geom = f.geom.clone();
            geom.set_spatial_ref(times_crs.clone());
            geom.transform_to_inplace(&ds_srs).ok()?;
            Some(geom)
        })
        .collect();
    let ret = unsafe {
        let geoms: Vec<OGRGeometryH> = geoms.iter().map(|x| x.c_geometry()).collect();
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

#[unsafe(no_mangle)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn propag_rasterize_fuels(
    fuels: *const FFIFuelFeature,
    fuels_len: usize,
    fuels_crs: *const c_char,
    geo_ref: &GeoReference,
    result: *mut u8,
    err_msg: *mut c_char,
    err_len: usize,
) -> bool {
    let fuels = if !fuels.is_null() && fuels_len > 0 {
        let is = std::slice::from_raw_parts(fuels, fuels_len);
        is.iter()
            .filter_map(|x| FuelFeature::try_from(x).ok())
            .collect()
    } else {
        Vec::new()
    };
    match (|| {
        let fuels_crs = spatial_ref_from_buf(fuels_crs)?;
        let vec = rasterize_fuels(&fuels, fuels_crs, geo_ref)?;
        std::ptr::copy_nonoverlapping(vec.as_ptr(), result, geo_ref.len() as usize);
        use gdal::errors::GdalError;
        Ok::<(), GdalError>(())
    })() {
        Ok(()) => true,
        Err(err) => {
            let err = format!("{}", err);
            if let Ok(c_err) = std::ffi::CString::new(err) {
                libc::strncpy(err_msg, c_err.as_ptr(), err_len);
            }
            false
        }
    }
}

pub fn rasterize_fuels(
    fuels: &[FuelFeature],
    fuels_crs: SpatialRef,
    geo_ref: &GeoReference,
) -> gdal::errors::Result<Vec<u8>> {
    let d = DriverManager::get_driver_by_name("MEM")?;
    let mut ds =
        d.create_with_band_type::<u8, _>("in-memory", geo_ref.width as _, geo_ref.height as _, 1)?;
    let ds_srs = crate::loader::to_spatial_ref(&geo_ref.proj)?;
    ds.set_spatial_ref(&ds_srs)?;
    ds.set_geo_transform(&geo_ref.transform.as_array_64())?;
    let mut b = ds.rasterband(1)?;
    let no_data: u8 = Max::MAX;
    b.fill(no_data as f64, None)?;
    b.set_no_data_value(Some(no_data as f64))?;
    let values: Vec<f64> = fuels.iter().map(|f| f.code as f64).collect();
    let geoms: Vec<Geometry> = fuels
        .iter()
        .filter_map(|f| {
            let mut geom = f.geom.clone();
            geom.set_spatial_ref(fuels_crs.clone());
            geom.transform_to_inplace(&ds_srs).ok()?;
            Some(geom)
        })
        .collect();
    let ret = unsafe {
        let geoms: Vec<OGRGeometryH> = geoms.iter().map(|x| x.c_geometry()).collect();
        gdal_sys::GDALRasterizeGeometries(
            ds.c_dataset(),
            1,
            [1i32].as_ptr(),
            geoms.len() as _,
            geoms.as_ptr(),
            None,
            core::ptr::null_mut(),
            values.as_ptr(),
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
