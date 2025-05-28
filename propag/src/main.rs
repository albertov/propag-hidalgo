use cust::prelude::*;
use firelib_rs::*;
use std::error::Error;
use uom::si::angle::degree;
use uom::si::f64::*;
use uom::si::length::foot;
use uom::si::ratio::ratio;
use uom::si::time::minute;
use uom::si::velocity::foot_per_minute;
use uom::si::velocity::meter_per_second;

#[macro_use]
extern crate timeit;

/// How many elems to generate
const NUMBERS_LEN: usize = 1_000_000;

static PTX: &str = include_str!("../../target/cuda/firelib.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    // generate our random vectors.
    use rand::prelude::*;
    let rng = &mut rand::rng();
    let mut mkratio = { || Ratio::new::<ratio>(rng.random_range(0..10000) as f64 / 10000.0) };
    let rng = &mut rand::rng();
    let mut azimuth = { || Angle::new::<degree>(rng.random_range(0..36000) as f64 / 100.0) };
    let rng = &mut rand::rng();
    let terrains: TerrainCudaVec = (0..NUMBERS_LEN)
        .map(|_n| {
            let wind_speed =
                Velocity::new::<meter_per_second>(rng.random_range(0..10000) as f64 / 1000.0);
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
    let models: Vec<usize> = (0..NUMBERS_LEN).map(|_n| rng.random_range(0..14)).collect();
    let mut fires: Vec<FireCuda> = std::iter::repeat(Fire::null().into())
        .take(NUMBERS_LEN)
        .collect();

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    println!("Gertting ctx");
    let _ctx = cust::quick_init()?;
    println!("Got ctx");

    let num_devices = Device::num_devices()?;
    println!("Number of devices: {}", num_devices);

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;
    println!("Got module");

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // retrieve the add kernel from the module so we can calculate the right launch config.
    let func = module.get_function("standard_burn")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (NUMBERS_LEN as u32).div_ceil(block_size);

    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );

    println!("Calculating with GPU");
    timeit!({
        // allocate the GPU memory needed to house our numbers and copy them over.
        let model_gpu = models.as_slice().as_dbuf()?;
        let d1hr_gpu = terrains.d1hr.as_slice().as_dbuf()?;
        let d10hr_gpu = terrains.d10hr.as_slice().as_dbuf()?;
        let d100hr_gpu = terrains.d100hr.as_slice().as_dbuf()?;
        let herb_gpu = terrains.herb.as_slice().as_dbuf()?;
        let wood_gpu = terrains.wood.as_slice().as_dbuf()?;
        let wind_speed_gpu = terrains.wind_speed.as_slice().as_dbuf()?;
        let wind_azimuth_gpu = terrains.wind_azimuth.as_slice().as_dbuf()?;
        let slope_gpu = terrains.slope.as_slice().as_dbuf()?;
        let aspect_gpu = terrains.aspect.as_slice().as_dbuf()?;

        // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
        // cost of the copy, but you need to be careful not to read from the buffer.
        let fires_buf = fires.as_slice().as_dbuf()?;

        // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
        // not block the thread until the kernel is finished.
        unsafe {
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
                    fires_buf.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        // copy back the data from the GPU.
        fires_buf.copy_to(&mut fires)?;
    });

    let terrains: Vec<Terrain> = terrains.iter().map(|t| t.into()).collect();
    let mut fires_rs: Vec<Fire> = vec![Fire::null(); NUMBERS_LEN];

    println!("Calculating with CPU");
    timeit!({
        fires_rs = models
            .iter()
            .zip(terrains.iter())
            .map(|(m, t)| {
                firelib_rs::Catalog::STANDARD
                    .get(*m)
                    .and_then(|f| f.burn(t))
                    .unwrap_or(Fire::null())
            })
            .collect()
    });
    assert!(fires.iter().zip(fires_rs.iter()).all(|(f_gpu, f_cpu)| {
        let f_gpu: Fire = (*f_gpu).into();
        f_gpu.almost_eq(f_cpu)
    }));

    Ok(())
}
