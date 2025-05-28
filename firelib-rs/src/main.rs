use firelib_rs::types::*;
use rand::prelude::*;
use uom::si::angle::degree;
use uom::si::f64::*;
use uom::si::ratio::ratio;
use uom::si::velocity::meter_per_second;

fn main() {
    let mut rng = rand::rng();
    let fuel = STANDARD_CATALOG.get(rng.random_range(0..14)).unwrap();
    let zero = Ratio::new::<ratio>(0.0);
    let zero_ms = Velocity::new::<meter_per_second>(0.0);
    let zero_deg = Angle::new::<degree>(0.0);
    let spread = fuel.spread(&Terrain {
        d1hr: zero,
        d10hr: zero,
        d100hr: zero,
        herb: zero,
        wood: zero,
        wind_speed: zero_ms,
        wind_azimuth: zero_deg,
        slope: zero,
        aspect: zero_deg,
    });
    println!(
        "fuel={}\nspread={:?}\nspeedAtAz={:?}",
        fuel.name,
        spread,
        spread.at_azimuth(Angle::new::<degree>(25.0)).speed()
    )
}
