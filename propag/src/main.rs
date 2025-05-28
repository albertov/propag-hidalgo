#![feature(int_roundings)]
use ::geometry::*;
use cust::function::{BlockSize, GridSize};
use cust::prelude::*;
use firelib_rs::float;
use firelib_rs::float::*;
use firelib_rs::*;
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

fn main() -> Result<(), Box<dyn Error>> {
    println!("Calculating with GPU Propag");
    let geo_ref: GeoReference = GeoReference::south_up(
        (
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 {
                x: 25000.0,
                y: 25000.0,
            },
        ),
        Vec2 { x: 25.0, y: 25.0 },
        25830,
    )
    .unwrap();
    let len = geo_ref.len();

    println!("Generating input data");
    type OptionalVec<T> = Vec<Option<T>>;
    let model: OptionalVec<usize> = (0..len).map(|_n| Some(1)).collect();
    let d1hr: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let d10hr: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let d100hr: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let herb: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let wood: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let wind_speed: OptionalVec<float::T> = (0..len).map(|_n| Some(1.0)).collect();
    let wind_azimuth: OptionalVec<float::T> = (0..len).map(|_n| Some(0.0)).collect();
    let aspect: OptionalVec<float::T> = (0..len).map(|_n| Some(0.0)).collect();
    let slope: OptionalVec<float::T> = (0..len).map(|_n| Some(PI)).collect();

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    println!("Loading module");
    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!("Getting function");
    // retrieve the add kernel from the module so we can calculate the right launch config.
    let propag = module.get_function("propag")?;
    let pre_burn = module.get_function("pre_burn")?;

    let grid_size = GridSize {
        x: geo_ref.size[0].div_ceil(THREAD_BLOCK_AXIS_LENGTH),
        y: geo_ref.size[1].div_ceil(THREAD_BLOCK_AXIS_LENGTH),
        z: 1,
    };
    let linear_grid_size: usize = (grid_size.x * grid_size.y * grid_size.z) as usize;

    let block_size = BlockSize {
        x: THREAD_BLOCK_AXIS_LENGTH,
        y: THREAD_BLOCK_AXIS_LENGTH,
        z: 1,
    };
    let linear_block_size = block_size.x * block_size.y * block_size.z;

    println!(
        "using grid_size={:?} blocks_size={:?} linear_grid_size={} for {} elems",
        grid_size, block_size, linear_grid_size, len
    );
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

    let (_, block_size2) = pre_burn.suggested_launch_configuration(0, 0.into())?;
    let grid_size2 = (model.len() as u32).div_ceil(block_size2) + 1;

    // input/output vectors
    let mut time: Vec<Option<u32>> = std::iter::repeat(None).take(model.len()).collect();
    let mut refs_x: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
    let mut refs_y: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
    let max_time: u32 = 60 * 60 * 50;
    let fire_pos = USizeVec2 { x: 500, y: 100 };

    ({
        let mut speed_max: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut azimuth_max: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut eccentricity: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut progress: Vec<u32> = std::iter::repeat(0).take(linear_grid_size).collect();

        time.fill(None);
        refs_x.fill(None);
        refs_y.fill(None);
        time[(fire_pos.x + fire_pos.y * geo_ref.size[0] as usize)] = Some(0);
        refs_x[(fire_pos.x + fire_pos.y * geo_ref.size[0] as usize)] = Some(fire_pos.x);
        refs_y[(fire_pos.x + fire_pos.y * geo_ref.size[0] as usize)] = Some(fire_pos.y);

        let mut speed_max_buf = speed_max.as_slice().as_dbuf()?;
        let mut azimuth_max_buf = azimuth_max.as_slice().as_dbuf()?;
        let mut eccentricity_buf = eccentricity.as_slice().as_dbuf()?;
        let mut refs_x_buf = refs_x.as_slice().as_dbuf()?;
        let mut refs_y_buf = refs_y.as_slice().as_dbuf()?;
        let mut time_buf = time.as_slice().as_dbuf()?;
        let mut progress_buf = progress.as_slice().as_dbuf()?;
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                pre_burn<<<grid_size2, block_size2, 0, stream>>>(
                    model_gpu.as_device_ptr(),
                    model_gpu.len(),
                    d1hr_gpu.as_device_ptr(),
                    d1hr_gpu.len(),
                    d10hr_gpu.as_device_ptr(),
                    d10hr_gpu.len(),
                    d100hr_gpu.as_device_ptr(),
                    d100hr_gpu.len(),
                    herb_gpu.as_device_ptr(),
                    herb_gpu.len(),
                    wood_gpu.as_device_ptr(),
                    wood_gpu.len(),
                    wind_speed_gpu.as_device_ptr(),
                    wind_speed_gpu.len(),
                    wind_azimuth_gpu.as_device_ptr(),
                    wind_azimuth_gpu.len(),
                    slope_gpu.as_device_ptr(),
                    slope_gpu.len(),
                    aspect_gpu.as_device_ptr(),
                    aspect_gpu.len(),
                    speed_max_buf.as_device_ptr(),
                    azimuth_max_buf.as_device_ptr(),
                    eccentricity_buf.as_device_ptr(),
                )
            )?;
            let width = geo_ref.size[0];
            stream.synchronize()?;
            loop {
                /*
                let num_times = time.iter().filter(|t| t.is_some()).count();
                azimuth_max_buf.copy_to(&mut azimuth_max)?;
                assert!(azimuth_max.iter().all(|t| t.is_some()));
                eccentricity_buf.copy_to(&mut eccentricity)?;
                assert!(eccentricity.iter().all(|t| t.is_some()));
                speed_max_buf.copy_to(&mut speed_max)?;
                assert!(speed_max.iter().all(|t| t.is_some()));
                */
                launch!(
                    // slices are passed as two parameters, the pointer and the length.
                    propag<<<grid_size, block_size, 0, stream>>>(
                        speed_max_buf.as_device_ptr(),
                        speed_max_buf.len(),
                        azimuth_max_buf.as_device_ptr(),
                        azimuth_max_buf.len(),
                        eccentricity_buf.as_device_ptr(),
                        eccentricity_buf.len(),
                        time_buf.as_device_ptr(),
                        refs_x_buf.as_device_ptr(),
                        refs_y_buf.as_device_ptr(),
                        progress_buf.as_device_ptr(),
                        geo_ref,
                        max_time,
                        linear_grid_size,
                    )
                )?;
                stream.synchronize()?;
                /*
                time_buf.copy_to(&mut time)?;
                refs_x_buf.copy_to(&mut refs_x)?;
                refs_y_buf.copy_to(&mut refs_y)?;
                assert!(
                    refs_x.iter().all(|x|x.map(|r|r==fire_pos.x).unwrap_or(true)),
                );
                assert!(
                    refs_y.iter().all(|x|x.map(|r|r==fire_pos.y).unwrap_or(true)),
                );
                assert_eq!(
                    refs_x.iter().filter(|x|x.is_some()).count(),
                    refs_y.iter().filter(|x|x.is_some()).count(),
                );
                let num_times_after = time.iter().filter(|t| t.is_some()).count();
                assert_eq!(
                    num_times_after,
                    refs_x.iter().filter(|x|x.is_some()).count(),
                );
                if num_times_after == num_times {
                    break;
                };
                println!("config_max_time={}", max_time);
                println!(
                    "max_time={:?}",
                    time.iter()
                        .filter_map(|x| *x)
                        .max()
                );
                */
                progress_buf.copy_to(&mut progress)?;
                println!(
                    "progress={:?}",
                    progress.as_slice().iter().filter(|p| **p > 0).count(),
                );
                if progress.iter().all(|x| *x == 0) {
                    break;
                }
            }
            time_buf.copy_to(&mut time)?;
        };
    });
    println!("config_max_time={}", max_time);
    println!("max_time={:?}", time.iter().filter_map(|x| *x).max());
    println!("min_time={:?}", time.iter().filter_map(|x| *x).min());
    let num_times_after = time.iter().filter(|t| t.is_some()).count();
    println!("num_times_after={}", num_times_after);
    //time_buf.copy_to(&mut time)?;
    //assert!(time.len() > 1);
    Ok(())
}
