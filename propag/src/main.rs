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
    // generate our random vectors.
    use rand::prelude::*;
    let rng = &mut rand::rng();
    let mut mkratio = { || Ratio::new::<ratio>(rng.random_range(0..10000) as float::T / 10000.0) };
    let rng = &mut rand::rng();
    let mut azimuth = { || Angle::new::<degree>(rng.random_range(0..36000) as float::T / 100.0) };
    let rng = &mut rand::rng();

    type OptionalVec<T> = Vec<Option<T>>;
    let model: OptionalVec<usize> = (0..len).map(|_n| Some(1)).collect();
    let d1hr: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let d10hr: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let d100hr: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let herb: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let wood: OptionalVec<float::T> = (0..len).map(|_n| Some(0.1)).collect();
    let wind_speed: OptionalVec<float::T> = (0..len).map(|_n| Some(5.0)).collect();
    let wind_azimuth: OptionalVec<float::T> = (0..len).map(|_n| Some(0.0)).collect();
    let aspect: OptionalVec<float::T> = (0..len).map(|_n| Some(PI)).collect();
    let slope: OptionalVec<float::T> = (0..len).map(|_n| Some(PI)).collect();

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // retrieve the add kernel from the module so we can calculate the right launch config.
    let propag = module.get_function("propag")?;
    let pre_burn = module.get_function("pre_burn")?;

    let grid_size = GridSize {
        x: geo_ref.size[0].div_ceil(THREAD_BLOCK_AXIS_LENGTH),
        y: geo_ref.size[1].div_ceil(THREAD_BLOCK_AXIS_LENGTH),
        z: 1,
    };
    let linear_grid_size: u32 = grid_size.x * grid_size.y * grid_size.z;

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

    timeit!({
        // input/output vectors
        let mut time: Vec<Option<float::T>> = std::iter::repeat(None).take(model.len()).collect();
        let mut refs_x: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
        let mut refs_y: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
        let mut out_time: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut out_refs_x: Vec<Option<usize>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut out_refs_y: Vec<Option<usize>> =
            std::iter::repeat(None).take(model.len()).collect();

        let mut speed_max: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut azimuth_max: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut eccentricity: Vec<Option<float::T>> =
            std::iter::repeat(None).take(model.len()).collect();
        let mut block_progress: Vec<float::T> = std::iter::repeat(0.0)
            .take(linear_grid_size as usize)
            .collect();

        let max_time: float::T = 60.0 * 60.0 * 20.0;
        time[(100 + 200 * geo_ref.size[0]) as usize] = Some(0.0);
        refs_x[(100 + 200 * geo_ref.size[0]) as usize] = Some(100);
        refs_y[(100 + 200 * geo_ref.size[0]) as usize] = Some(200);

        let mut speed_max_buf = speed_max.as_slice().as_dbuf()?;
        let mut azimuth_max_buf = azimuth_max.as_slice().as_dbuf()?;
        let mut eccentricity_buf = eccentricity.as_slice().as_dbuf()?;
        let mut refs_x_buf = refs_x.as_slice().as_dbuf()?;
        let mut refs_y_buf = refs_y.as_slice().as_dbuf()?;
        let mut time_buf = time.as_slice().as_dbuf()?;
        let mut out_refs_x_buf = out_refs_x.as_slice().as_dbuf()?;
        let mut out_refs_y_buf = out_refs_y.as_slice().as_dbuf()?;
        let mut out_time_buf = out_time.as_slice().as_dbuf()?;
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
            let times: Vec<(u32, u32, f32)> = (0..len)
                .zip(time.iter())
                .filter_map(|(i, t)| Some((i % width, i.div_floor(width), (*t)?)))
                .collect();
            stream.synchronize()?;
            loop {
                let num_times = time.iter().filter(|t| t.is_some()).count();
                launch!(
                    // slices are passed as two parameters, the pointer and the length.
                    propag<<<grid_size, block_size, 0, stream>>>(
                        geo_ref,
                        max_time,
                        speed_max_buf.as_device_ptr(),
                        speed_max_buf.len(),
                        azimuth_max_buf.as_device_ptr(),
                        azimuth_max_buf.len(),
                        eccentricity_buf.as_device_ptr(),
                        eccentricity_buf.len(),
                        time_buf.as_device_ptr(),
                        refs_x_buf.as_device_ptr(),
                        refs_y_buf.as_device_ptr(),
                    )
                )?;
                stream.synchronize()?;
                time_buf.copy_to(&mut time)?;
                let num_times_after = time.iter().filter(|t| t.is_some()).count();
                println!("num_times_after={}", num_times_after);
                let width = geo_ref.size[0];
                if num_times_after == num_times {
                    break;
                }
            }
        };
    });
    //time_buf.copy_to(&mut time)?;
    //assert!(time.len() > 1);
    Ok(())
}
