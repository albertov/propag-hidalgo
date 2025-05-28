#![feature(int_roundings)]
use ::geometry::*;
use cust::device::DeviceAttribute;
use cust::function::{BlockSize, GridSize};
use cust::memory::UnifiedBox;
use cust::prelude::*;
use firelib_cuda::{Settings, HALO_RADIUS, SIZEOF_FBC_SHARED_ITEM};
use firelib_rs::float;
use firelib_rs::float::*;
use firelib_rs::*;
use gdal::raster::*;
use gdal::spatial_ref::SpatialRef;
use gdal::vector::*;
use gdal::*;
use min_max_traits::Max;
use num_traits::Float;
use std::error::Error;
use uom::si::angle::degree;
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

#[macro_use]
extern crate timeit;

mod loader;
const THREAD_BLOCK_AXIS_LENGTH: u32 = 16;

static PTX: &str = include_str!("../../target/cuda/firelib.ptx");
static PTX_C: &str = include_str!("../../target/cuda/propag_c.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    println!("Calculating with GPU Propag",);
    let max_time: f32 = 60.0 * 60.0 * 10.0;
    let geo_ref: GeoReference = GeoReference::south_up(
        (
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 {
                x: 5000.0,
                y: 5000.0,
            },
        ),
        Vec2 { x: 1.0, y: 1.0 },
        25830,
    )
    .unwrap();
    let fire_pos = USizeVec2 {
        x: geo_ref.width as usize / 2 - THREAD_BLOCK_AXIS_LENGTH as usize / 2,
        y: geo_ref.height as usize / 2 - THREAD_BLOCK_AXIS_LENGTH as usize / 2,
    };
    println!("fire_pos={:?}", fire_pos);
    let len = geo_ref.len();

    println!("Generating input data");
    type OptionalVec<T> = Vec<Option<T>>;
    let mut model: Vec<usize> = (0..len).map(|_n| 1).collect();
    let d1hr: Vec<float::T> = (0..len).map(|_n| 0.1).collect();
    let d10hr: Vec<float::T> = (0..len).map(|_n| 0.1).collect();
    let d100hr: Vec<float::T> = (0..len).map(|_n| 0.1).collect();
    let herb: Vec<float::T> = (0..len).map(|_n| 0.1).collect();
    let wood: Vec<float::T> = (0..len).map(|_n| 0.1).collect();
    let wind_speed: Vec<float::T> = (0..len).map(|_n| 5.0).collect();
    let wind_azimuth: Vec<float::T> = (0..len).map(|_n| 0.0).collect();
    let aspect: Vec<float::T> = (0..len).map(|_n| 0.0).collect();
    let slope: Vec<float::T> = (0..len).map(|_n| 0.0).collect();

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    cust::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let ctx = Context::new(device)?;
    ctx.set_flags(ContextFlags::SCHED_AUTO)?;

    println!("Loading module");
    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;
    let module_c = Module::from_ptx(PTX_C, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!("Getting function");
    // retrieve the add kernel from the module so we can calculate the right launch config.
    let propag_c = module_c.get_function("propag")?;
    let pre_burn = module.get_function("standard_simple_burn")?;
    let find_boundary_changes = module.get_function("find_boundary_changes")?;

    let block_size = BlockSize {
        x: THREAD_BLOCK_AXIS_LENGTH,
        y: THREAD_BLOCK_AXIS_LENGTH,
        z: 1,
    };

    let linear_block_size = block_size.x * block_size.y * block_size.z;
    let radius = HALO_RADIUS as u32;
    let shmem_size = ((block_size.x + radius * 2) * (block_size.y + radius * 2));
    let shmem_bytes = shmem_size * 48; //FIXME: std::mem::size_of::<Point>() as u32;
                                       //assert_eq!(std::mem::size_of::<Point>(), 64);

    let max_active_blocks =
        propag_c.max_active_blocks_per_multiprocessor(block_size, shmem_bytes as _)?;
    println!("max_active_blocks_per_multiprocessor={}", max_active_blocks);
    let mp_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
    let max_total_blocks = mp_count as u32 * max_active_blocks;
    println!("max_total_blocks={}", max_total_blocks);

    // Assumes square block size
    assert_eq!(block_size.x, block_size.y);
    let grid_w = (max_total_blocks as f64).sqrt().floor() as u32;
    let grid_size: GridSize = (grid_w, grid_w).into();

    let super_grid_size: (u32, u32) = (
        geo_ref.width.div_ceil(grid_size.x * block_size.x),
        geo_ref.height.div_ceil(grid_size.y * block_size.y),
    );

    println!(
        "using geo_ref={:?}\ngrid_size={:?}\nblocks_size={:?}\nsuper_grid_size={:?}\nfor {} elems",
        geo_ref, grid_size, block_size, super_grid_size, len
    );
    let lo: i32 = -30;
    for i in (lo..30) {
        model[fire_pos.x + (i as usize) + (fire_pos.y + 5) * geo_ref.width as usize] = 0;
        model[fire_pos.x + (i as usize) + (fire_pos.y + 6) * geo_ref.width as usize] = 0;
        model[fire_pos.x + (i as usize) + (fire_pos.y + 7) * geo_ref.width as usize] = 0;
    }
    // allocate the GPU memory needed to house our numbers and copy them over.
    let model_gpu = model.as_slice().as_dbuf()?;
    let d1hr_gpu = d1hr.as_slice().as_dbuf()?;
    let d10hr_gpu = d10hr.as_slice().as_dbuf()?;
    let d100hr_gpu = d100hr.as_slice().as_dbuf()?;
    let herb_gpu = herb.as_slice().as_dbuf()?;
    let wood_gpu = wood.as_slice().as_dbuf()?;
    let wind_speed_gpu = wind_speed.as_slice().as_dbuf()?;
    let wind_azimuth_gpu = wind_azimuth.as_slice().as_dbuf()?;
    let slope_gpu = slope.as_slice().as_dbuf()?;
    let aspect_gpu = aspect.as_slice().as_dbuf()?;

    // input/output vectors
    let mut time: Vec<f32> = std::iter::repeat(Max::MAX).take(model.len()).collect();
    let mut refs_x: Vec<u16> = std::iter::repeat(Max::MAX).take(model.len()).collect();
    let mut refs_y: Vec<u16> = std::iter::repeat(Max::MAX).take(model.len()).collect();
    let mut boundary_change: Vec<u16> = std::iter::repeat(0).take(model.len()).collect();

    let find_boundaries_grid_size = GridSize {
        x: geo_ref.width.div_ceil(block_size.x),
        y: geo_ref.height.div_ceil(block_size.y),
        z: 1,
    };

    timeit!({
        let mut speed_max: Vec<float::T> = std::iter::repeat(0.0).take(model.len()).collect();
        let mut azimuth_max: Vec<float::T> = std::iter::repeat(0.0).take(model.len()).collect();
        let mut eccentricity: Vec<float::T> = std::iter::repeat(0.0).take(model.len()).collect();

        time.fill(Max::MAX);
        refs_x.fill(Max::MAX);
        refs_y.fill(Max::MAX);
        time[(fire_pos.x + fire_pos.y * geo_ref.width as usize)] = 0.0;
        refs_x[(fire_pos.x + fire_pos.y * geo_ref.width as usize)] = fire_pos.x as u16;
        refs_y[(fire_pos.x + fire_pos.y * geo_ref.width as usize)] = fire_pos.y as u16;

        let mut speed_max_buf = speed_max.as_slice().as_dbuf()?;
        let mut azimuth_max_buf = azimuth_max.as_slice().as_dbuf()?;
        let mut eccentricity_buf = eccentricity.as_slice().as_dbuf()?;
        let mut refs_x_buf = refs_x.as_slice().as_dbuf()?;
        let mut refs_y_buf = refs_y.as_slice().as_dbuf()?;
        let mut time_buf = time.as_slice().as_dbuf()?;
        let mut boundary_change_buf = boundary_change.as_slice().as_dbuf()?;
        let (_, pre_burn_block_size) = pre_burn.suggested_launch_configuration(0, 0.into())?;
        let pre_burn_grid_size = geo_ref.len().div_ceil(pre_burn_block_size);
        unsafe {
            println!("pre burn");
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                pre_burn<<<pre_burn_grid_size, pre_burn_block_size, 0, stream>>>(
                    model.len(),
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
            //println!("propag");
            let settings = Settings { geo_ref, max_time };
            println!("propag");
            loop {
                let mut worked: Vec<UnifiedBox<u32>> =
                    Vec::with_capacity((super_grid_size.0 * super_grid_size.1) as usize);
                for grid_x in (0..super_grid_size.0) {
                    for grid_y in (0..super_grid_size.1) {
                        let linear_grid_size: usize =
                            (grid_size.x * grid_size.y * grid_size.z) as usize;
                        let mut progress: Vec<u32> =
                            std::iter::repeat(0).take(linear_grid_size).collect();
                        let mut progress_buf = progress.as_slice().as_dbuf()?;
                        let this_worked = UnifiedBox::new(0)?;
                        cust::launch_cooperative!(
                            // slices are passed as two parameters, the pointer and the length.
                            propag_c<<<grid_size, block_size, shmem_bytes, stream>>>(
                                settings,
                                grid_x,
                                grid_y,
                                this_worked.as_unified_ptr(),
                                speed_max_buf.as_device_ptr(),
                                azimuth_max_buf.as_device_ptr(),
                                eccentricity_buf.as_device_ptr(),
                                time_buf.as_device_ptr(),
                                refs_x_buf.as_device_ptr(),
                                refs_y_buf.as_device_ptr(),
                                boundary_change_buf.as_device_ptr(),
                                progress_buf.as_device_ptr(),
                            )
                        )?;
                        worked.push(this_worked);
                    }
                }
                stream.synchronize()?;
                if worked.iter().all(|x| **x == 0) {
                    break;
                }
            }
            stream.synchronize()?;
            println!("find boundary_change done");
            /*
            refs_x_buf.copy_to(&mut refs_x)?;
            refs_y_buf.copy_to(&mut refs_y)?;
            assert!(refs_x.iter().all(|x| *x == Max::MAX || *x == fire_pos.x),);
            assert!(refs_y.iter().all(|x| *x == Max::MAX || *x == fire_pos.y),);
            assert_eq!(
                refs_x.iter().filter(|x| *x < &Max::MAX).count(),
                refs_y.iter().filter(|x| *x < &Max::MAX).count(),
            );
            */
        };
        time_buf.copy_to(&mut time)?;
        boundary_change_buf.copy_to(&mut boundary_change)?;
        refs_x_buf.copy_to(&mut refs_x)?;
        refs_y_buf.copy_to(&mut refs_y)?;
    });
    let good_times: Vec<f32> = time
        .iter()
        .filter_map(|x| if *x < Max::MAX { Some(*x) } else { None })
        .collect();
    println!("config_max_time={}", max_time);
    println!(
        "max_time={:?}",
        good_times.iter().max_by(|a, b| a.total_cmp(b))
    );
    println!(
        "max_time={:?}",
        good_times.iter().min_by(|a, b| a.total_cmp(b))
    );
    let num_times_after = good_times.iter().count();
    println!("num_times_after={}", num_times_after);
    //time_buf.copy_to(&mut time)?;
    //assert!(time.len() > 1);

    // Write times raster
    println!("Generating times geotiff");
    let gtiff = DriverManager::get_driver_by_name("GTIFF")?;
    let options = RasterCreationOptions::from_iter(["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"]);
    let mut ds = gtiff.create_with_band_type_with_options::<f32, _>(
        "tiempos.tif",
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
    let mut buf = Buffer::new(band.size(), time);
    band.write((0, 0), band.size(), &mut buf)?;

    // Write refs vectors
    println!("Generating refs shapefile");
    let shape = DriverManager::get_driver_by_name("ESRI Shapefile")?;
    let mut ds = shape.create_vector_only("refs")?;
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
