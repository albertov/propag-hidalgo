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

/// How many elems to generate
const NUMBERS_LEN: usize = 1_000_000;

static PTX: &str = include_str!("../../target/cuda/firelib.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    // generate our random vectors.
    use rand::prelude::*;
    let rng = &mut rand::rng();
    let mut mkratio = { || Ratio::new::<ratio>(rng.random_range(0..10000) as float::T / 10000.0) };
    let rng = &mut rand::rng();
    let mut azimuth = { || Angle::new::<degree>(rng.random_range(0..36000) as float::T / 100.0) };
    let rng = &mut rand::rng();
    type OptionalVec<T> = Vec<Option<T>>;
    let model: OptionalVec<usize> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..14)))
        .collect();
    let d1hr: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 10000.0))
        .collect();
    let d10hr: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 10000.0))
        .collect();
    let d100hr: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 10000.0))
        .collect();
    let herb: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 10000.0))
        .collect();
    let wood: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 10000.0))
        .collect();
    let wind_speed: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 1000.0))
        .collect();
    let wind_azimuth: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..720) as float::T / 4.0 * PI))
        .collect();
    let aspect: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..720) as float::T / 4.0 * PI))
        .collect();
    let slope: OptionalVec<float::T> = (0..NUMBERS_LEN)
        .map(|_n| Some(rng.random_range(0..10000) as float::T / 10000.0))
        .collect();

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
    let func = module.get_function("pre_burn")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (model.len() as u32).div_ceil(block_size);

    println!(
        "using {} blocks and {} threads per block for {} elems",
        grid_size,
        block_size,
        model.len()
    );

    println!("Calculating with GPU");
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

    // output vectors
    let mut speed_max: OptionalVec<float::T> = std::iter::repeat(None).take(model.len()).collect();
    let mut azimuth_max: OptionalVec<float::T> =
        std::iter::repeat(None).take(model.len()).collect();
    let mut eccentricity: OptionalVec<float::T> =
        std::iter::repeat(None).take(model.len()).collect();
    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        // allocate our output buffers. You could also use DeviceBuffer::uninitialized() to avoid the
        // cost of the copy, but you need to be careful not to read from the buffer.
        let mut speed_max_buf = speed_max.as_slice().as_dbuf()?;
        let mut azimuth_max_buf = azimuth_max.as_slice().as_dbuf()?;
        let mut eccentricity_buf = eccentricity.as_slice().as_dbuf()?;
        timeit!({
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
                    speed_max_buf.as_device_ptr(),
                    azimuth_max_buf.as_device_ptr(),
                    eccentricity_buf.as_device_ptr(),
                )
            )?;
            stream.synchronize()?;
        });

        // copy back the data from the GPU.
        speed_max_buf.copy_to(&mut speed_max)?;
        azimuth_max_buf.copy_to(&mut azimuth_max)?;
        eccentricity_buf.copy_to(&mut eccentricity)?;
    }

    let fire: Vec<Option<FireSimple>> = (0..model.len())
        .map(|i| {
            Some(
                (FireSimpleCuda {
                    speed_max: speed_max[i]?,
                    azimuth_max: azimuth_max[i]?,
                    eccentricity: eccentricity[i]?,
                })
                .into(),
            )
        })
        .collect();

    let terrain: Vec<Option<Terrain>> = (0..model.len())
        .map(|i| {
            Some(
                (TerrainCuda {
                    d1hr: d1hr[i]?,
                    d10hr: d10hr[i]?,
                    d100hr: d100hr[i]?,
                    herb: herb[i]?,
                    wood: wood[i]?,
                    wind_speed: wind_speed[i]?,
                    wind_azimuth: wind_azimuth[i]?,
                    slope: slope[i]?,
                    aspect: aspect[i]?,
                })
                .into(),
            )
        })
        .collect();
    let mut fire_rs: Vec<Option<FireSimple>> = Vec::new();
    for _ in 0..model.len() {
        fire_rs.push(None)
    }

    println!("Calculating with CPU");
    timeit!({
        fire_rs = model
            .iter()
            .zip(terrain.iter())
            .map(|(m, t)| firelib_rs::Catalog::STANDARD.burn_simple((*m)?, &(*t)?))
            .collect()
    });
    println!("Verifying results");
    assert!(fire
        .iter()
        .zip(fire_rs.iter())
        .map(|(f_gpu, f_cpu)| {
            let f_gpu = Into::<FireSimple>::into((*f_gpu)?);
            let f_cpu = (*f_cpu)?;
            let res = f_gpu.almost_eq(&f_cpu);
            if !res {
                println!("{:?} /= {:?}", f_gpu, f_cpu);
            }
            Some(res)
        })
        .all(|x| x != Some(false)));
    assert!(fire_rs.iter().any(|f| f.is_some()));
    assert!(fire_rs.iter().any(|f| f.is_none()));
    println!("All equal");

    Ok(())
}
