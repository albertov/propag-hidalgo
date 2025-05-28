#[cfg(feature = "std")]
fn main() {
    use firelib_rs::float::*;
    use firelib_rs::*;
    use rand::prelude::*;
    use uom::si::angle::degree;
    use uom::si::ratio::ratio;
    use uom::si::velocity::meter_per_second;

    let mut rng = rand::rng();
    let fuel = Catalog::STANDARD.get(rng.random_range(1..14)).unwrap();
    let zero = Ratio::new::<ratio>(0.0);
    let zero_ms = Velocity::new::<meter_per_second>(0.0);
    let zero_deg = Angle::new::<degree>(0.0);
    if let Some(fire) = fuel.burn(&Terrain {
        d1hr: zero,
        d10hr: zero,
        d100hr: zero,
        herb: zero,
        wood: zero,
        wind_speed: zero_ms,
        wind_azimuth: zero_deg,
        slope: zero,
        aspect: zero_deg,
    }) {
        println!(
            "fuel={}\nfire={:?}\nspeedAtAz={:?}",
            std::str::from_utf8(&fuel.name).unwrap(),
            fire,
            fire.spread(Angle::new::<degree>(25.0)).speed()
        )
    }
}
