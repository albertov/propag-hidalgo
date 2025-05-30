#![no_std]
#![feature(test)]

#[cfg(not(target_os = "cuda"))]
extern crate std;

#[macro_use]
extern crate uom;

#[cfg(not(target_os = "cuda"))]
#[macro_use]
extern crate soa_derive;

pub mod cuda;
pub mod firelib;

mod f32;
pub mod float {
    #[doc(inline)]
    pub use super::f32::*;
}

#[macro_use]
pub mod units;

pub use crate::cuda::*;
pub use crate::firelib::*;

impl Catalog {
    pub const STANDARD: Catalog = Catalog::make([
        Fuel::standard(b"NoFuel", b"No Combustible Fuel", 0.1, 0.01, []),
        Fuel::standard(
            b"NFFL01",
            b"Short Grass (1 ft)",
            1.0,
            0.12,
            [ParticleDef::standard(ParticleType::Dead, 0.0340, 3500.0)],
        ),
        Fuel::standard(
            b"NFFL02",
            b"Timber (grass & understory)",
            1.0,
            0.15,
            [
                ParticleDef::standard(ParticleType::Dead, 0.0920, 3000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0460, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0230, 30.0),
                ParticleDef::standard(ParticleType::Herb, 0.0230, 1500.0),
            ],
        ),
        Fuel::standard(
            b"NFFL03",
            b"Tall Grass (2.5 ft)",
            2.5,
            0.25,
            [ParticleDef::standard(ParticleType::Dead, 0.1380, 1500.0)],
        ),
        Fuel::standard(
            b"NFFL04",
            b"Chaparral (6 ft)",
            6.0,
            0.2,
            [
                ParticleDef::standard(ParticleType::Dead, 0.2300, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.1840, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0920, 30.0),
                ParticleDef::standard(ParticleType::Wood, 0.2300, 1500.0),
            ],
        ),
        Fuel::standard(
            b"NFFL05",
            b"Brush (2 ft)",
            2.0,
            0.2,
            [
                ParticleDef::standard(ParticleType::Dead, 0.0460, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0230, 109.0),
                ParticleDef::standard(ParticleType::Wood, 0.0920, 1500.0),
            ],
        ),
        Fuel::standard(
            b"NFFL06",
            b"Dormant Brush & Hardwood Slash",
            2.5,
            0.25,
            [
                ParticleDef::standard(ParticleType::Dead, 0.0690, 1750.0),
                ParticleDef::standard(ParticleType::Dead, 0.1150, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0920, 30.0),
            ],
        ),
        Fuel::standard(
            b"NFFL07",
            b"Southern Rough",
            2.5,
            0.40,
            [
                ParticleDef::standard(ParticleType::Dead, 0.0520, 1750.0),
                ParticleDef::standard(ParticleType::Dead, 0.0860, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0690, 30.0),
                ParticleDef::standard(ParticleType::Wood, 0.0170, 1550.0),
            ],
        ),
        Fuel::standard(
            b"NFFL08",
            b"Closed Timber Litter",
            0.2,
            0.30,
            [
                ParticleDef::standard(ParticleType::Dead, 0.0690, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0460, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.1150, 30.0),
            ],
        ),
        Fuel::standard(
            b"NFFL09",
            b"Hardwood Litter",
            0.2,
            0.25,
            [
                ParticleDef::standard(ParticleType::Dead, 0.1340, 2500.0),
                ParticleDef::standard(ParticleType::Dead, 0.0190, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0070, 30.0),
            ],
        ),
        Fuel::standard(
            b"NFFL10",
            b"Timber (litter & understory)",
            1.0,
            0.25,
            [
                ParticleDef::standard(ParticleType::Dead, 0.1380, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0920, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.2300, 30.0),
                ParticleDef::standard(ParticleType::Wood, 0.0920, 1500.0),
            ],
        ),
        Fuel::standard(
            b"NFFL11",
            b"Light Logging Slash",
            1.0,
            0.15,
            [
                ParticleDef::standard(ParticleType::Dead, 0.0690, 1500.0),
                ParticleDef::standard(ParticleType::Dead, 0.2070, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.2530, 30.0),
            ],
        ),
        Fuel::standard(
            b"NFFL12",
            b"Medium Logging Slash",
            2.3,
            0.20,
            [
                ParticleDef::standard(ParticleType::Dead, 0.1840, 1500.0),
                ParticleDef::standard(ParticleType::Dead, 0.6440, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.7590, 30.0),
            ],
        ),
        Fuel::standard(
            b"NFFL13",
            b"Heavy Logging Slash",
            3.0,
            0.25,
            [
                ParticleDef::standard(ParticleType::Dead, 0.3220, 1500.0),
                ParticleDef::standard(ParticleType::Dead, 0.0580, 109.0),
                ParticleDef::standard(ParticleType::Dead, 1.2880, 30.0),
            ],
        ),
    ]);
}

#[cfg(test)]
mod tests {
    use super::float;
    use super::float::*;
    use super::*;
    use crate::units::linear_power_density::btu_foot_sec;
    use firelib_sys::*;
    use std::ffi::CString;
    use std::vec::Vec;
    use uom::si::angle::degree;
    use uom::si::length::foot;

    use crate::units::heat_flux_density::btu_sq_foot_min;
    use crate::units::radiant_exposure::btu_sq_foot;
    use uom::si::ratio::ratio;
    use uom::si::time::minute;
    use uom::si::velocity::foot_per_minute;
    use uom::si::velocity::meter_per_second;

    extern crate quickcheck;

    use quickcheck::{Arbitrary, Gen};
    extern crate test;
    use test::Bencher;

    quickcheck::quickcheck! {
        fn we_produce_the_same_output_as_firelib_c(terrain: Terrain, azimuth: ValidAzimuth) -> bool {
            let az = Angle::new::<degree>(azimuth.0);
            let (firelib_sp, firelib_sp_az) = firelib_rs_spread(&terrain, az);
            let (c_sp, c_sp_az) = firelib_c_spread(&terrain, az);
            Fire::almost_eq(&firelib_sp, &c_sp) && SpreadAtAzimuth::almost_eq(&firelib_sp_az, &c_sp_az)
        }
    }
    fn firelib_rs_spread(terrain: &Terrain, azimuth: Angle) -> (Fire, SpreadAtAzimuth) {
        if let Some(fire) = Catalog::STANDARD.burn(terrain) {
            let spread_az = SpreadAtAzimuth::from_spread(&fire.spread(azimuth));
            (fire, spread_az)
        } else {
            (Fire::NULL, SpreadAtAzimuth::null())
        }
    }

    fn firelib_c_spread(terrain: &Terrain, azimuth: Angle) -> (Fire, SpreadAtAzimuth) {
        unsafe {
            let name = CString::new("standard").unwrap();
            let catalog = Fire_FuelCatalogCreateStandard(name.as_ptr(), 13);
            let mut m = [
                terrain.d1hr.get::<ratio>().into(),
                terrain.d10hr.get::<ratio>().into(),
                terrain.d100hr.get::<ratio>().into(),
                (0.0).into(),
                terrain.herb.get::<ratio>().into(),
                terrain.wood.get::<ratio>().into(),
            ];
            Fire_SpreadNoWindNoSlope(catalog, terrain.fuel_code as _, m.as_mut_ptr());
            Fire_SpreadWindSlopeMax(
                catalog,
                terrain.fuel_code as _,
                terrain.wind_speed.get::<foot_per_minute>().into(),
                terrain.wind_azimuth.get::<degree>().into(),
                terrain.slope.get::<ratio>().into(),
                terrain.aspect.get::<degree>().into(),
            );
            let fuel: *mut fuelModelDataStruct =
                *(*catalog).modelPtr.wrapping_add(terrain.fuel_code as _);
            Fire_SpreadAtAzimuth(
                catalog,
                terrain.fuel_code as _,
                (*fuel).azimuthMax,
                (FIRE_BYRAMS | FIRE_FLAME).try_into().unwrap(),
            );
            let f = *fuel;
            let fire = Fire {
                rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(f.rxInt as float::T),
                speed0: Velocity::new::<foot_per_minute>(f.spread0 as float::T),
                hpua: RadiantExposure::new::<btu_sq_foot>(f.hpua as float::T),
                phi_eff_wind: Ratio::new::<ratio>(f.phiEw as float::T),
                speed_max: Velocity::new::<foot_per_minute>(f.spreadMax as float::T),
                azimuth_max: Angle::new::<degree>(f.azimuthMax as float::T),
                eccentricity: Ratio::new::<ratio>(f.eccentricity as float::T),
                residence_time: Time::new::<minute>(f.taur as float::T),
            };
            Fire_SpreadAtAzimuth(
                catalog,
                terrain.fuel_code as _,
                azimuth.get::<degree>().into(),
                (FIRE_BYRAMS | FIRE_FLAME).try_into().unwrap(),
            );
            let f = *fuel;
            let spread = SpreadAtAzimuth {
                speed: Velocity::new::<foot_per_minute>(f.spreadAny as float::T),
                byrams: LinearPowerDensity::new::<btu_foot_sec>(f.byrams as float::T),
                flame: Length::new::<foot>(f.flame as float::T),
            };
            Fire_FuelCatalogDestroy(catalog);
            (fire, spread)
        }
    }

    #[bench]
    fn bench_firelib_rs(b: &mut Bencher) {
        bench_spread_fn(b, firelib_rs_spread)
    }

    #[bench]
    fn bench_firelib_c(b: &mut Bencher) {
        bench_spread_fn(b, firelib_c_spread)
    }

    fn bench_spread_fn(b: &mut Bencher, f: impl Fn(&Terrain, Angle) -> (Fire, SpreadAtAzimuth)) {
        use rand::prelude::*;
        let rng = &mut rand::rng();
        let mut mkratio =
            { || Ratio::new::<ratio>(rng.random_range(0..10000) as float::T / 10000.0) };
        let rng = &mut rand::rng();
        let mut azimuth =
            { || Angle::new::<degree>(rng.random_range(0..36000) as float::T / 100.0) };
        let rng = &mut rand::rng();
        let args: Vec<(Terrain, Angle)> = (0..1000)
            .map(|_n| {
                let wind_speed = Velocity::new::<meter_per_second>(
                    rng.random_range(0..30000) as float::T / 1000.0,
                );
                let terrain = Terrain {
                    fuel_code: rng.random_range(0..13),
                    d1hr: mkratio(),
                    d10hr: mkratio(),
                    d100hr: mkratio(),
                    herb: mkratio(),
                    wood: mkratio(),
                    wind_speed,
                    wind_azimuth: azimuth(),
                    slope: mkratio(),
                    aspect: azimuth(),
                };
                (terrain, azimuth())
            })
            .collect();
        b.iter(|| {
            args.iter()
                .map(|(terrain, az)| f(terrain, *az).1.speed.get::<meter_per_second>())
                .sum::<float::T>()
        })
    }

    impl Arbitrary for Terrain {
        fn arbitrary(g: &mut Gen) -> Self {
            let fuel_codes = &(0..13).collect::<Vec<_>>()[..];
            let fuel_code = *g.choose(fuel_codes).unwrap();
            let humidities = &(0..101).map(|x| x as float::T / 100.0).collect::<Vec<_>>()[..];
            let wind_speeds = &(0..101).map(|x| x as float::T / 10.0).collect::<Vec<_>>()[..];
            let slopes = &(0..1001)
                .map(|x| x as float::T / 1000.0)
                .collect::<Vec<_>>()[..];
            let mut choose_humidity = || Ratio::new::<ratio>(*g.choose(humidities).unwrap());
            Self {
                fuel_code,
                d1hr: choose_humidity(),
                d10hr: choose_humidity(),
                d100hr: choose_humidity(),
                herb: choose_humidity(),
                wood: choose_humidity(),
                wind_speed: Velocity::new::<meter_per_second>(*g.choose(wind_speeds).unwrap()),
                wind_azimuth: Angle::new::<degree>(ValidAzimuth::arbitrary(g).0),
                slope: Ratio::new::<ratio>(*g.choose(slopes).unwrap()),
                aspect: Angle::new::<degree>(ValidAzimuth::arbitrary(g).0),
            }
        }
    }

    #[derive(Debug, Clone)]
    struct ValidAzimuth(float::T);

    impl Arbitrary for ValidAzimuth {
        fn arbitrary(g: &mut Gen) -> Self {
            let azimuths = &(0..3601).map(|x| x as float::T / 10.0).collect::<Vec<_>>()[..];
            Self(*g.choose(azimuths).unwrap())
        }
    }

    #[derive(Debug)]
    struct SpreadAtAzimuth {
        speed: Velocity,
        byrams: LinearPowerDensity,
        flame: Length,
    }

    impl SpreadAtAzimuth {
        fn from_spread(s: &Spread<'_, Fire>) -> Self {
            SpreadAtAzimuth {
                flame: s.flame(),
                speed: s.speed(),
                byrams: s.byrams(),
            }
        }
        fn null() -> SpreadAtAzimuth {
            SpreadAtAzimuth {
                speed: Velocity::new::<foot_per_minute>(0.0),
                byrams: LinearPowerDensity::new::<btu_foot_sec>(0.0),
                flame: Length::new::<foot>(0.0),
            }
        }
        fn almost_eq(&self, other: &Self) -> bool {
            fuzzy_cmp(
                "speed",
                self.speed.get::<foot_per_minute>(),
                other.speed.get::<foot_per_minute>(),
            ) && fuzzy_cmp(
                "byrams",
                self.byrams.get::<btu_foot_sec>(),
                other.byrams.get::<btu_foot_sec>(),
            ) && fuzzy_cmp("flame", self.flame.get::<foot>(), other.flame.get::<foot>())
        }
    }
}
