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
    let geo_ref : GeoReference = GeoReference::south_up(
        (Vec2{x:0.0,y:0.0},Vec2{x:25000.0,y:25000.0}),
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
    let terrain: TerrainCudaVec = (0..len)
        .map(|_n| {
            let wind_speed =
                Velocity::new::<meter_per_second>(1.0);
            From::from(Terrain {
                d1hr: Ratio::new::<ratio>(0.1),
                d10hr: Ratio::new::<ratio>(0.1),
                d100hr: Ratio::new::<ratio>(0.1),
                herb: Ratio::new::<ratio>(0.1),
                wood: Ratio::new::<ratio>(0.1),
                wind_speed,
                wind_azimuth: Angle::new::<degree>(0.0),
                slope: Ratio::new::<ratio>(0.0),
                aspect: Angle::new::<degree>(180.0),
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
        x: geo_ref.size[0] / THREAD_BLOCK_AXIS_LENGTH + 1,
        y: geo_ref.size[1] / THREAD_BLOCK_AXIS_LENGTH + 1,
        z: 1,
    };
    let linear_grid_size : u32 = grid_size.x * grid_size.y * grid_size.z;

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
    let mut time: Vec<Option<float::T>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut speed_max: Vec<Option<float::T>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut azimuth_max: Vec<Option<float::T>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut eccentricity: Vec<Option<float::T>> = std::iter::repeat(None)
        .take(model.len())
        .collect();
    let mut refs_x: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
    let mut refs_y: Vec<Option<usize>> = std::iter::repeat(None).take(model.len()).collect();
    let mut block_progress: Vec<float::T> = std::iter::repeat(0.0)
        .take(linear_grid_size as usize)
        .collect();

    let max_time = (60.0*60.0*500.0 as float::T);
    time[(500 + 200 * geo_ref.size[0]) as usize] = Some(0.0);
    refs_x[(500 + 200 * geo_ref.size[0]) as usize] = Some(500);
    refs_y[(500 + 200 * geo_ref.size[0]) as usize] = Some(200);

    let mut speed_max_buf = speed_max.as_slice().as_unified_buf()?;
    let mut azimuth_max_buf = azimuth_max.as_slice().as_unified_buf()?;
    let mut eccentricity_buf = eccentricity.as_slice().as_unified_buf()?;
    let mut refs_x_buf = refs_x.as_slice().as_unified_buf()?;
    let mut refs_y_buf = refs_y.as_slice().as_unified_buf()?;
    let mut time_buf = time.as_slice().as_unified_buf()?;
    let mut block_progress_buf = block_progress.as_slice().as_unified_buf()?;

    unsafe {
        ({
            loop {
                let num_times = time_buf
                    .as_slice()
                    .iter()
                    .filter(|t|t.is_some())
                    .count();
                println!("num_times={}", num_times);
                let num_fires = speed_max_buf
                    .as_slice()
                    .iter()
                    .filter(|t|t.is_some())
                    .count();
                println!("num_fires={}", num_fires);
                let num_refs = refs_x_buf
                    .as_slice()
                    .iter()
                    .filter(|t|t.is_some())
                    .count();
                println!("num_refs={}", num_refs);
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                func<<<grid_size, block_size, 0, stream>>>(
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
                    speed_max_buf.as_unified_ptr(),
                    azimuth_max_buf.as_unified_ptr(),
                    eccentricity_buf.as_unified_ptr(),
                    time_buf.as_unified_ptr(),
                    refs_x_buf.as_unified_ptr(),
                    refs_y_buf.as_unified_ptr(),
                    block_progress_buf.as_unified_ptr(),
                    geo_ref,
                    linear_grid_size,
                    max_time,
                )
            )?;
            stream.synchronize()?;
            let num_times_after = time_buf
                .as_slice()
                .iter()
                .filter(|t|t.is_some())
                .count();
            if num_times_after == num_times {
                break
            }

            }
        });

    }
    assert!(time_buf.as_slice().iter().filter(|t| t.is_some()).count() > 1);
    let time : Vec<f32> =
        time_buf.as_slice().iter().filter_map(|t| *t).collect();
    println!("{:?}", time);
    Ok(())
}
