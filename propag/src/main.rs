use cust::prelude::*;
use firelib_rs::float;
use firelib_rs::float::*;
use firelib_rs::*;
use std::error::Error;
use uom::si::angle::degree;
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

#[macro_use]
extern crate timeit;

mod loader;

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
    let terrain: TerrainCudaVec = (0..NUMBERS_LEN)
        .map(|_n| {
            let wind_speed =
                Velocity::new::<meter_per_second>(rng.random_range(0..10000) as float::T / 1000.0);
            From::from(Terrain {
                d1hr: mkratio(),
                d10hr: mkratio(),
                d100hr: mkratio(),
                herb: mkratio(),
                wood: mkratio(),
                wind_speed,
                wind_azimuth: azimuth(),
                slope: mkratio(),
                aspect: azimuth(),
            })
        })
        .collect();
    let model: Vec<usize> = (0..NUMBERS_LEN).map(|_n| rng.random_range(0..14)).collect();

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
    let func = module.get_function("standard_burn")?;

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
    let d1hr_gpu = terrain.d1hr.as_slice().as_dbuf()?;
    let d10hr_gpu = terrain.d10hr.as_slice().as_dbuf()?;
    let d100hr_gpu = terrain.d100hr.as_slice().as_dbuf()?;
    let herb_gpu = terrain.herb.as_slice().as_dbuf()?;
    let wood_gpu = terrain.wood.as_slice().as_dbuf()?;
    let wind_speed_gpu = terrain.wind_speed.as_slice().as_dbuf()?;
    let wind_azimuth_gpu = terrain.wind_azimuth.as_slice().as_dbuf()?;
    let slope_gpu = terrain.slope.as_slice().as_dbuf()?;
    let aspect_gpu = terrain.aspect.as_slice().as_dbuf()?;

    // output vectors
    let mut fire: FireCudaVec = std::iter::repeat(Fire::NULL.into())
        .take(model.len())
        .collect();
    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        // allocate our output buffers. You could also use DeviceBuffer::uninitialized() to avoid the
        // cost of the copy, but you need to be careful not to read from the buffer.
        let rx_int_buf = DeviceBuffer::uninitialized(model.len())?;
        let speed0_buf = DeviceBuffer::uninitialized(model.len())?;
        let hpua_buf = DeviceBuffer::uninitialized(model.len())?;
        let phi_eff_wind_buf = DeviceBuffer::uninitialized(model.len())?;
        let speed_max_buf = DeviceBuffer::uninitialized(model.len())?;
        let azimuth_max_buf = DeviceBuffer::uninitialized(model.len())?;
        let eccentricity_buf = DeviceBuffer::uninitialized(model.len())?;
        let residence_time_buf = DeviceBuffer::uninitialized(model.len())?;
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
                    rx_int_buf.as_device_ptr(),
                    speed0_buf.as_device_ptr(),
                    hpua_buf.as_device_ptr(),
                    phi_eff_wind_buf.as_device_ptr(),
                    speed_max_buf.as_device_ptr(),
                    azimuth_max_buf.as_device_ptr(),
                    eccentricity_buf.as_device_ptr(),
                    residence_time_buf.as_device_ptr(),
                )
            )?;
            stream.synchronize()?;
        });

        // copy back the data from the GPU.
        rx_int_buf.copy_to(&mut fire.rx_int)?;
        speed0_buf.copy_to(&mut fire.speed0)?;
        hpua_buf.copy_to(&mut fire.hpua)?;
        phi_eff_wind_buf.copy_to(&mut fire.phi_eff_wind)?;
        speed_max_buf.copy_to(&mut fire.speed_max)?;
        azimuth_max_buf.copy_to(&mut fire.azimuth_max)?;
        eccentricity_buf.copy_to(&mut fire.eccentricity)?;
        residence_time_buf.copy_to(&mut fire.residence_time)?;
    }

    let terrain: Vec<Terrain> = terrain.iter().map(|t| t.into()).collect();
    let mut fire_rs: Vec<Fire> = Vec::new();
    for _ in 0..fire.len() {
        fire_rs.push(Fire::NULL)
    }

    println!("Calculating with CPU");
    timeit!({
        fire_rs = model
            .iter()
            .zip(terrain.iter())
            .map(|(m, t)| firelib_rs::Catalog::STANDARD.burn(*m, t))
            .map(|f| f.unwrap_or(Fire::NULL))
            .collect()
    });
    println!("Verifying results");
    assert!(fire
        .iter()
        .zip(fire_rs.iter())
        .all(|(f_gpu, f_cpu)| Into::<Fire>::into(f_gpu).almost_eq(f_cpu)));
    println!("All equal");

    println!("Calculating with GPU Simple");
    // output vectors
    let mut fire_simple: FireSimpleCudaVec = std::iter::repeat(FireSimple::NULL.into())
        .take(model.len())
        .collect();
    unsafe {
        // allocate our output buffers. You could also use DeviceBuffer::uninitialized() to avoid the
        // cost of the copy, but you need to be careful not to read from the buffer.
        let speed_max_buf = DeviceBuffer::uninitialized(model.len())?;
        let azimuth_max_buf = DeviceBuffer::uninitialized(model.len())?;
        let eccentricity_buf = DeviceBuffer::uninitialized(model.len())?;
        let func = module.get_function("standard_simple_burn")?;
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
        speed_max_buf.copy_to(&mut fire_simple.speed_max)?;
        azimuth_max_buf.copy_to(&mut fire_simple.azimuth_max)?;
        eccentricity_buf.copy_to(&mut fire_simple.eccentricity)?;
    }

    println!("Calculating with CPU Simple");
    let mut fire_simple_rs: Vec<FireSimple> = Vec::new();
    for _ in 0..fire.len() {
        fire_simple_rs.push(FireSimple::NULL)
    }
    timeit!({
        fire_simple_rs = model
            .iter()
            .zip(terrain.iter())
            .map(|(m, t)| firelib_rs::Catalog::STANDARD.burn_simple(*m, t))
            .map(|f| f.unwrap_or(FireSimple::NULL))
            .collect()
    });
    println!("Verifying results");
    assert!(fire_simple
        .iter()
        .zip(fire_simple_rs.iter())
        .all(|(f_gpu, f_cpu)| Into::<FireSimple>::into(f_gpu).almost_eq(f_cpu)));
    println!("All equal");
    Ok(())
}
