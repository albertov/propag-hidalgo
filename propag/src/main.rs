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
    let geo_ref = GeoReference::<f64>::south_up(
        Rect::new(
            Coord { x: 0.0, y: 0.0 },
            Coord {
                x: 1000.0,
                y: 1000.0,
            },
        ),
        Coord { x: 1.0, y: 1.0 },
        Crs::Epsg(25830),
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
    let terrain: TerrainCudaVec = (0..len)
        .map(|_n| {
            let wind_speed =
                Velocity::new::<meter_per_second>(rng.random_range(0..10000) as float::T / 1000.0);
            From::from(Terrain {
                d1hr: Ratio::new::<ratio>(0.0),
                d10hr: Ratio::new::<ratio>(0.0),
                d100hr: Ratio::new::<ratio>(0.0),
                herb: Ratio::new::<ratio>(0.0),
                wood: Ratio::new::<ratio>(0.0),
                wind_speed,
                wind_azimuth: azimuth(),
                slope: mkratio(),
                aspect: azimuth(),
            })
        })
        .collect();
    let model: Vec<usize> = (0..len).map(|_n| 1).collect();

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
    let func = module.get_function("propag")?;

    let grid_size = GridSize {
        x: geo_ref.size[0].div_ceil(THREAD_BLOCK_AXIS_LENGTH),
        y: geo_ref.size[1].div_ceil(THREAD_BLOCK_AXIS_LENGTH),
        z: 1,
    };
    let linear_grid_size = grid_size.x * grid_size.y * grid_size.z;

    let block_size = BlockSize {
        x: THREAD_BLOCK_AXIS_LENGTH,
        y: THREAD_BLOCK_AXIS_LENGTH,
        z: 1,
    };
    let linear_block_size = block_size.x * block_size.y * block_size.z;

    println!(
        "using grid_size={:?} blocks_size={:?} linear_grid_size={} for {} elems",
        grid_size,
        block_size,
        linear_grid_size,
        len
    );
    // allocate the GPU memory needed to house our numbers and copy them over.
    let model_gpu = model.as_slice().as_dbuf()?;
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
    let mut time: Vec<Option<f32>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    time[model.len().div_ceil(2) + 200] = Some(0.0);
    let mut speed_max: Vec<Option<f32>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut azimuth_max: Vec<Option<f32>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut eccentricity: Vec<Option<f32>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut refs_x: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
    let mut refs_y: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
    let mut block_progress: Vec<f32> = std::iter::repeat(0.0)
        .take(linear_grid_size as usize)
        .collect();

    let max_time = (60.0*60.0*5.0 as float::T);
    unsafe {
        let speed_max_buf = speed_max.as_slice().as_dbuf()?;
        let azimuth_max_buf = azimuth_max.as_slice().as_dbuf()?;
        let eccentricity_buf = eccentricity.as_slice().as_dbuf()?;
        let refs_x_buf = refs_x.as_slice().as_dbuf()?;
        let refs_y_buf = refs_y.as_slice().as_dbuf()?;
        let time_buf = time.as_slice().as_dbuf()?;
        let block_progress_buf = block_progress.as_slice().as_dbuf()?;

        timeit!({
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                func<<<grid_size, block_size, 0, stream>>>(
                    geo_ref,
                    linear_grid_size,
                    max_time,
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
                    time_buf.as_device_ptr(),
                    refs_x_buf.as_device_ptr(),
                    refs_y_buf.as_device_ptr(),
                    block_progress_buf.as_device_ptr(),
                )
            )?;
            stream.synchronize()?;
        });

        // copy back the data from the GPU.
        time_buf.copy_to(&mut time)?;
        refs_x_buf.copy_to(&mut refs_x)?;
        refs_y_buf.copy_to(&mut refs_y)?;
        block_progress_buf.copy_to(&mut block_progress)?;
    }
    assert_eq!(block_progress, (0..linear_grid_size)
        .map(|_| 0.0)
        .collect::<Vec<f32>>());

    assert!(time.iter().filter(|t| t.is_some()).count() > 1);
    Ok(())
}
