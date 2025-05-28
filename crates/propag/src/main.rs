#![feature(int_roundings)]
use ::geometry::*;
use firelib::TerrainCuda;
use gdal::vector::*;
use min_max_traits::Max;
use propag::loader::to_spatial_ref;
use propag::propag::Settings;
use propag::propag::{propagate, Propagation, TimeFeature};
use std::error::Error;

#[macro_use]
extern crate timeit;

fn main() -> Result<(), Box<dyn Error>> {
    let max_time: f32 = 60.0 * 60.0 * 10.0;
    let px = Vec2 { x: 5.0, y: 5.0 };
    let geo_ref: GeoReference = GeoReference::south_up(
        (
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 {
                x: px.x * 1024.0,
                y: px.y * 1024.0,
            },
        ),
        px,
        "+proj=utm +zone=30 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
    )
    .unwrap();
    let settings = Settings { geo_ref, max_time };
    let fire_pos = USizeVec2 {
        x: geo_ref.width as usize / 2 - 24_usize / 2,
        y: geo_ref.height as usize / 2 - 24_usize / 2,
    };
    let fire_pos = geo_ref.backward(fire_pos);
    let wkt = format!("POINT({} {})", fire_pos.x, fire_pos.y);
    let propag = Propagation {
        settings,
        output_path: "tiempos.tif".to_string(),
        refs_output_path: None,
        block_boundaries_out_path: None,
        grid_boundaries_out_path: None,
        initial_ignited_elements: vec![TimeFeature {
            time: 0.0,
            geom: Geometry::from_wkt(&wkt)?,
        }],
        initial_ignited_elements_crs: to_spatial_ref(&geo_ref.proj)?,
        terrain_loader: Box::new(TerrainCuda {
            fuel_code: 1,
            d1hr: 0.1,
            d10hr: 0.1,
            d100hr: 0.1,
            herb: 0.1,
            wood: 0.1,
            wind_speed: 5.0,
            wind_azimuth: 0.0,
            aspect: 0.0,
            slope: 0.0,
        }),
    };

    timeit!({
        let result = propagate(&propag)?;
        let good_times: Vec<f32> = result
            .time
            .iter()
            .filter_map(|x| if *x < Max::MAX { Some(*x) } else { None })
            .collect();
        println!("config_max_time={}", settings.max_time);
        println!(
            "max_time={:?}",
            good_times.iter().max_by(|a, b| a.total_cmp(b))
        );
        println!(
            "max_time={:?}",
            good_times.iter().min_by(|a, b| a.total_cmp(b))
        );
        let num_times_after = good_times.len();
        println!("num_times_after={}", num_times_after);
    });
    Ok(())
}
