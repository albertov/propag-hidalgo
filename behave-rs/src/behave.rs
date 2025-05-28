use crate::units::areal_mass_density::pound_per_square_foot;
use crate::units::heat_flux_density::btu_sq_foot_min;
use crate::units::linear_power_density::btu_foot_sec;
use crate::units::radiant_exposure::btu_sq_foot;
use crate::units::reciprocal_length::reciprocal_foot;
use uom::si::angle::degree;
use uom::si::available_energy::btu_per_pound;
use uom::si::f64::*;
use uom::si::length::foot;
use uom::si::mass_density::pound_per_cubic_foot;
use uom::si::ratio::ratio;
use uom::si::velocity::foot_per_minute;

use crate::types::*;

const SMIDGEN: f64 = 1e-6;

impl Spread {
    pub fn no_spread() -> Spread {
        Spread {
            rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(0.0),
            speed0: Velocity::new::<foot_per_minute>(0.0),
            hpua: RadiantExposure::new::<btu_sq_foot>(0.0),
            phi_eff_wind: Ratio::new::<ratio>(0.0),
            speed_max: Velocity::new::<foot_per_minute>(0.0),
            azimuth_max: Angle::new::<degree>(0.0),
            eccentricity: Ratio::new::<ratio>(0.0),
            byrams_max: LinearPowerDensity::new::<btu_foot_sec>(0.0),
            flame_max: Length::new::<foot>(0.0),
        }
    }

    pub fn at_azimuth(&self, azimuth: Angle) -> SpreadAtAzimuth {
        // TODO
        SpreadAtAzimuth::no_spread()
    }
}

impl SpreadAtAzimuth {
    pub fn no_spread() -> SpreadAtAzimuth {
        SpreadAtAzimuth {
            speed: Velocity::new::<foot_per_minute>(0.0),
            byrams: LinearPowerDensity::new::<btu_foot_sec>(0.0),
            flame: Length::new::<foot>(0.0),
        }
    }
}

impl ParticleDef {
    pub fn standard(p_type: ParticleType, p_load: f64, p_savr: f64) -> ParticleDef {
        ParticleDef {
            type_: p_type,
            load: ArealMassDensity::new::<pound_per_square_foot>(p_load),
            savr: ReciprocalLength::new::<reciprocal_foot>(p_savr),
            density: MassDensity::new::<pound_per_cubic_foot>(32.0),
            heat: AvailableEnergy::new::<btu_per_pound>(8000.0),
            si_total: Ratio::new::<ratio>(0.0555),
            si_effective: Ratio::new::<ratio>(0.01),
        }
    }
}
fn safe_div(a: f64, b: f64) -> f64 {
    if b > SMIDGEN {
        a / b
    } else {
        0.0
    }
}

impl Particle {
    pub fn make(def: &ParticleDef) -> Particle {
        let ParticleDef {
            type_,
            load,
            savr,
            density,
            heat,
            si_total,
            si_effective,
        } = def;
        Particle {
            type_: *type_,
            load: load.get::<pound_per_square_foot>(),
            savr: savr.get::<reciprocal_foot>(),
            density: density.get::<pound_per_cubic_foot>(),
            heat: heat.get::<btu_per_pound>(),
            si_total: si_total.get::<ratio>(),
            si_effective: si_effective.get::<ratio>(),
        }
    }
    const fn life(&self) -> Life {
        match &self.type_ {
            ParticleType::Dead => Life::Dead,
            _ => Life::Alive,
        }
    }

    fn moisture(&self, terrain: &Terrain) -> f64 {
        (match self.type_ {
            ParticleType::Herb => terrain.herb,
            ParticleType::Wood => terrain.wood,
            ParticleType::Dead => match self.size_class() {
                SizeClass::SC0 | SizeClass::SC1 => terrain.d1hr,
                SizeClass::SC2 | SizeClass::SC3 => terrain.d10hr,
                SizeClass::SC4 | SizeClass::SC5 => terrain.d100hr,
            },
        })
        .get::<ratio>()
    }

    fn surface_area(&self) -> f64 {
        safe_div(self.load * self.savr, self.density)
    }

    fn sigma_factor(&self) -> f64 {
        safe_div(-138.0, self.savr).exp()
    }

    fn size_class(&self) -> SizeClass {
        let savr = self.savr;
        if savr > 1200.0 {
            SizeClass::SC0
        } else if savr > 192.0 {
            SizeClass::SC1
        } else if savr > 96.0 {
            SizeClass::SC2
        } else if savr > 48.0 {
            SizeClass::SC3
        } else if savr > 16.0 {
            SizeClass::SC4
        } else {
            SizeClass::SC5
        }
    }
}

impl Fuel {
    pub fn make(def: FuelDef) -> Fuel {
        let (alive_particles, dead_particles) = def
            .particles
            .iter()
            .map(|p| Particle::make(p))
            .partition(move |p| p.life() == Life::Alive);
        Fuel {
            name: def.name,
            desc: def.desc,
            depth: def.depth.get::<foot>(),
            mext: def.mext.get::<ratio>(),
            adjust: def.adjust.get::<ratio>(),
            alive_particles,
            dead_particles,
        }
    }
    fn particles(&self) -> impl Iterator<Item = &Particle> {
        self.dead_particles
            .iter()
            .chain(self.alive_particles.iter())
    }

    fn life_particles(&self, life: Life) -> impl Iterator<Item = &Particle> {
        match life {
            Life::Alive => self.alive_particles.iter(),
            Life::Dead => self.dead_particles.iter(),
        }
    }
    fn has_live_particles(&self) -> bool {
        !self.alive_particles.is_empty()
    }
    fn total_area(&self) -> f64 {
        self.particles().map(|p| p.surface_area()).sum()
    }
    fn life_area_weight(&self, life: Life) -> f64 {
        self.life_particles(life)
            .map(|p| p.surface_area() / self.total_area())
            .sum()
    }
    fn life_fine_load(&self, life: Life) -> f64 {
        match life {
            Life::Alive => self
                .life_particles(life)
                .map(|p| p.load * (-500.0 / p.savr).exp())
                .sum(),
            Life::Dead => self
                .life_particles(life)
                .map(|p| p.load * p.sigma_factor())
                .sum(),
        }
    }
    fn life_load(&self, life: Life) -> f64 {
        self.life_particles(life)
            .map(|p| self.part_size_class_weight(p) * p.load * (1.0 - p.si_total))
            .sum()
    }
    fn life_savr(&self, life: Life) -> f64 {
        self.life_particles(life)
            .map(|p| self.part_area_weight(p) * p.savr)
            .sum()
    }
    fn life_heat(&self, life: Life) -> f64 {
        self.life_particles(life)
            .map(|p| self.part_area_weight(p) * p.heat)
            .sum()
    }
    fn life_seff(&self, life: Life) -> f64 {
        self.life_particles(life)
            .map(|p| self.part_area_weight(p) * p.si_effective)
            .sum()
    }
    fn life_eta_s(&self, life: Life) -> f64 {
        let seff: f64 = self.life_seff(life);
        if seff > SMIDGEN {
            let eta = 0.174 / seff.powf(0.19);
            if eta < 1.0 {
                eta
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
    fn sigma(&self) -> f64 {
        self.life_area_weight(Life::Alive) * self.life_savr(Life::Alive)
            + self.life_area_weight(Life::Dead) * self.life_savr(Life::Dead)
    }
    fn ratio(&self) -> f64 {
        self.beta() / (3.348 / self.sigma().powf(0.8189))
    }
    fn flux_ratio(&self) -> f64 {
        let sigma = self.sigma();
        let beta = self.beta();
        ((0.792 + 0.681 * sigma.sqrt()) * (beta + 0.1)).exp() / (192.0 + 0.2595 * sigma)
    }
    fn beta(&self) -> f64 {
        safe_div(
            self.particles().map(|p| safe_div(p.load, p.density)).sum(),
            self.depth,
        )
    }
    fn gamma(&self) -> f64 {
        let sigma = self.sigma();
        let sigma15 = sigma.powf(1.5);
        let gamma_max = sigma15 / (495.0 + 0.0594 * sigma15);
        let aa = 133.0 / sigma.powf(0.7913);
        gamma_max * self.ratio().powf(aa) * (aa * (1.0 - self.ratio())).exp()
    }

    fn life_rx_factor(&self, life: Life) -> f64 {
        self.life_load(life) * self.life_heat(life) * self.life_eta_s(life) * self.gamma()
    }
    fn part_area_weight(&self, particle: &Particle) -> f64 {
        safe_div(
            particle.surface_area(),
            self.life_particles(particle.life())
                .map(|p| p.surface_area())
                .sum(),
        )
    }

    fn part_size_class_weight(&self, particle: &Particle) -> f64 {
        self.life_particles(particle.life())
            .filter(|p| p.size_class() == particle.size_class())
            .map(|p| self.part_area_weight(p))
            .sum()
    }

    fn live_ext_factor(&self) -> f64 {
        2.9 * safe_div(
            self.life_fine_load(Life::Dead),
            self.life_fine_load(Life::Alive),
        )
    }

    fn bulk_density(&self) -> f64 {
        self.particles().map(|p| safe_div(p.load, self.depth)).sum()
    }

    fn residence_time(&self) -> f64 {
        384.0 / self.sigma()
    }

    fn slope_k(&self) -> f64 {
        5.275 * self.beta().powf(-0.3)
    }

    fn wind_bke(&self) -> (f64, f64, f64) {
        let sigma = self.sigma();
        let wind_b = 0.02526 * sigma.powf(0.54);
        let r = self.ratio();
        let c = 7.47 * ((-0.133) * (sigma.powf(0.55))).exp();
        let e = 0.715 * ((-0.000359) * sigma).exp();
        let wind_k = c * r.powf(-e);
        let wind_e = r.powf(e) / c;
        (wind_b, wind_k, wind_e)
    }

    pub fn combustion(&self) -> Combustion {
        let (wind_b, wind_k, wind_e) = self.wind_bke();
        Combustion {
            fuel: self.clone(),
            live_area_weight: self.life_area_weight(Life::Alive),
            live_rx_factor: self.life_rx_factor(Life::Alive),
            dead_area_weight: self.life_area_weight(Life::Dead),
            dead_rx_factor: self.life_rx_factor(Life::Dead),
            fine_dead_factor: self.life_fine_load(Life::Dead),
            live_ext_factor: self.live_ext_factor(),
            fuel_bed_bulk_dens: self.bulk_density(),
            residence_time: self.residence_time(),
            flux_ratio: self.flux_ratio(),
            slope_k: self.slope_k(),
            wind_b,
            wind_e,
            wind_k,
        }
    }
}

impl Combustion {
    pub fn spread(&self, terrain: &Terrain) -> Spread {
        if self.fuel.alive_particles.is_empty() && self.fuel.dead_particles.is_empty() {
            Spread::no_spread()
        } else {
            Spread {
                rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(self.rx_int(terrain)),
                speed0: Velocity::new::<foot_per_minute>(self.speed0(terrain)),
                hpua: RadiantExposure::new::<btu_sq_foot>(self.hpua(terrain)),
                phi_eff_wind: Ratio::new::<ratio>(self.phi_eff_wind(terrain)),
                speed_max: Velocity::new::<foot_per_minute>(self.speed_max(terrain)),
                azimuth_max: Angle::new::<degree>(self.azimuth_max(terrain)),
                eccentricity: Ratio::new::<ratio>(self.eccentricity(terrain)),
                byrams_max: LinearPowerDensity::new::<btu_foot_sec>(self.byrams_max(terrain)),
                flame_max: Length::new::<foot>(self.flame_max(terrain)),
            }
        }
    }
    fn rx_int(&self, terrain: &Terrain) -> f64 {
        self.fuel.life_rx_factor(Life::Alive) * self.life_eta_m(Life::Alive, terrain)
            + self.fuel.life_rx_factor(Life::Dead) * self.life_eta_m(Life::Dead, terrain)
    }
    fn speed0(&self, terrain: &Terrain) -> f64 {
        safe_div(self.rx_int(terrain) * self.flux_ratio, self.rbqig(terrain))
    }
    fn hpua(&self, terrain: &Terrain) -> f64 {
        self.rx_int(terrain) * self.residence_time
    }
    fn phi_eff_wind(&self, terrain: &Terrain) -> f64 {
        //todo!()
        0.0
    }
    fn speed_max(&self, terrain: &Terrain) -> f64 {
        self.speed0(terrain) * (1.0 + self.phi_ew(terrain))
    }
    fn azimuth_max(&self, terrain: &Terrain) -> f64 {
        //todo!()
        0.0
    }
    fn eccentricity(&self, terrain: &Terrain) -> f64 {
        //todo!()
        0.0
    }
    fn byrams_max(&self, terrain: &Terrain) -> f64 {
        self.residence_time * self.speed_max(terrain) * self.rx_int(terrain) / 60.0
    }
    fn flame_max(&self, terrain: &Terrain) -> f64 {
        flame_length(self.byrams_max(terrain))
    }

    fn life_moisture(&self, life: Life, terrain: &Terrain) -> f64 {
        self.fuel
            .life_particles(life)
            .map(|p| self.fuel.part_area_weight(p) * p.moisture(terrain))
            .sum()
    }

    fn life_mext(&self, life: Life, terrain: &Terrain) -> f64 {
        match life {
            Life::Alive => {
                if self.fuel.has_live_particles() {
                    let fdmois = safe_div(self.wfmd(terrain), self.fine_dead_factor);
                    let live_mext = self.live_ext_factor * (1.0 - fdmois / self.fuel.mext) - 0.226;
                    live_mext.max(self.fuel.mext)
                } else {
                    0.0
                }
            }
            Life::Dead => self.fuel.mext,
        }
    }

    fn life_eta_m(&self, life: Life, terrain: &Terrain) -> f64 {
        if self.life_moisture(life, terrain) >= self.life_mext(life, terrain) {
            0.0
        } else if self.life_mext(life, terrain) > SMIDGEN {
            let rt = self.life_moisture(life, terrain) / self.life_mext(life, terrain);
            1.0 - 2.59 * rt + 5.11 * rt * rt - 3.52 * rt * rt * rt
        } else {
            0.0
        }
    }

    fn wfmd(&self, terrain: &Terrain) -> f64 {
        self.fuel
            .life_particles(Life::Dead)
            .map(|p| p.moisture(terrain) * p.sigma_factor() * p.load)
            .sum()
    }

    fn rbqig(&self, terrain: &Terrain) -> f64 {
        let x: f64 = self
            .fuel
            .particles()
            .map(|p| {
                (250.0 + 1116.0 * p.moisture(terrain))
                    * self.fuel.part_area_weight(p)
                    * self.fuel.life_area_weight(p.life())
                    * p.sigma_factor()
            })
            .sum();
        self.fuel_bed_bulk_dens * x
    }

    fn phi_slope(&self, terrain: &Terrain) -> f64 {
        self.slope_k * terrain.slope.get::<ratio>().powf(2.0)
    }

    fn phi_wind(&self, terrain: &Terrain) -> f64 {
        if terrain.wind_speed.get::<foot_per_minute>() > SMIDGEN {
            let ws = terrain.wind_speed.get::<foot_per_minute>();
            self.wind_k * ws.powf(self.wind_b)
        } else {
            0.0
        }
    }

    fn phi_ew(&self, terrain: &Terrain) -> f64 {
        self.phi_slope(terrain) + self.phi_wind(terrain)
    }
}

fn flame_length(byrams: f64) -> f64 {
    if byrams > SMIDGEN {
        0.45 * byrams.powf(0.46)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use firelib_sys::*;
    use std::ffi::CString;

    extern crate quickcheck;

    use quickcheck::{Arbitrary, Gen};

    fn almost_eq(a: f64, b: f64) -> bool {
        let r = (a - b).abs() < SMIDGEN;
        println!("{:?} = {:?}", a, b);
        r
    }
    #[test]
    fn particles_works() {
        assert_eq!(STANDARD_CATALOG.get(4).unwrap().particles().count(), 4)
    }

    quickcheck::quickcheck! {
        fn behave_rs_produces_same_output_as_firelib(terrain: Terrain, model: ValidModel, azimuth: ValidAzimuth) -> bool {
            let (behave_sp, _behave_sp_az) = behave_rs_spread(model.0, &terrain, azimuth.0);
            let (firelib_sp, _firelib_sp_az) = firelib_spread(model.0, &terrain, azimuth.0);
            almost_eq(
                behave_sp.speed0.get::<foot_per_minute>(),
                firelib_sp.speed0.get::<foot_per_minute>(),
            )
        }
    }

    fn behave_rs_spread(
        model: usize,
        terrain: &Terrain,
        azimuth: f64,
    ) -> (Spread, SpreadAtAzimuth) {
        match STANDARD_CATALOG.get(model) {
            Some(fuel) => {
                let spread = fuel.combustion().spread(terrain);
                let spread_az = spread.at_azimuth(Angle::new::<degree>(azimuth));
                (spread, spread_az)
            }
            None => (Spread::no_spread(), SpreadAtAzimuth::no_spread()),
        }
    }

    fn firelib_spread(model: usize, terrain: &Terrain, azimuth: f64) -> (Spread, SpreadAtAzimuth) {
        unsafe {
            let name = CString::new("standard").unwrap();
            let catalog = Fire_FuelCatalogCreateStandard(name.as_ptr(), 20);
            let mut m = [
                terrain.d1hr.get::<ratio>(),
                terrain.d10hr.get::<ratio>(),
                terrain.d100hr.get::<ratio>(),
                (0.0).into(),
                terrain.herb.get::<ratio>(),
                terrain.wood.get::<ratio>(),
            ];
            Fire_SpreadNoWindNoSlope(catalog, model, m.as_mut_ptr());
            Fire_SpreadWindSlopeMax(
                catalog,
                model,
                terrain.wind_speed.get::<foot_per_minute>(),
                terrain.wind_azimuth.get::<degree>(),
                terrain.slope.into(),
                terrain.aspect.get::<degree>(),
            );
            let fuel: *mut fuelModelDataStruct = *(*catalog).modelPtr.wrapping_add(model);
            let f = *fuel;
            Fire_SpreadAtAzimuth(
                catalog,
                model,
                azimuth,
                (FIRE_BYRAMS | FIRE_FLAME).try_into().unwrap(),
            );
            let spread = Spread {
                rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(f.rxInt),
                speed0: Velocity::new::<foot_per_minute>(f.spread0),
                hpua: RadiantExposure::new::<btu_sq_foot>(f.hpua),
                phi_eff_wind: Ratio::new::<ratio>(f.phiEw),
                speed_max: Velocity::new::<foot_per_minute>(f.spreadMax),
                azimuth_max: Angle::new::<degree>(f.azimuthMax),
                eccentricity: Ratio::new::<ratio>(f.eccentricity),
                byrams_max: LinearPowerDensity::new::<btu_foot_sec>(f.byrams),
                flame_max: Length::new::<foot>(f.flame),
            };
            let spread_az = SpreadAtAzimuth {
                speed: Velocity::new::<foot_per_minute>(f.spreadAny),
                byrams: LinearPowerDensity::new::<btu_foot_sec>(f.byrams),
                flame: Length::new::<foot>(f.flame),
            };
            Fire_FuelCatalogDestroy(catalog);
            (spread, spread_az)
        }
    }
    use uom::si::velocity::meter_per_second;
    impl Arbitrary for Terrain {
        fn arbitrary(g: &mut Gen) -> Self {
            let humidities = &(0..101).map(|x| x as f64 / 100.0).collect::<Vec<_>>()[..];
            let wind_speeds = &(0..101).map(|x| x as f64 / 10.0).collect::<Vec<_>>()[..];
            let slopes = &(0..1001).map(|x| x as f64 / 1000.0).collect::<Vec<_>>()[..];
            let mut choose_humidity = || Ratio::new::<ratio>(*g.choose(humidities).unwrap());
            Self {
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
    struct ValidAzimuth(f64);

    impl Arbitrary for ValidAzimuth {
        fn arbitrary(g: &mut Gen) -> Self {
            let azimuths = &(0..3601).map(|x| x as f64 / 10.0).collect::<Vec<_>>()[..];
            Self(*g.choose(azimuths).unwrap())
        }
    }

    #[derive(Debug, Clone)]
    struct ValidModel(usize);

    impl Arbitrary for ValidModel {
        fn arbitrary(g: &mut Gen) -> Self {
            let models = &(0..7).collect::<Vec<_>>()[..]; //FIXME
            Self(*g.choose(models).unwrap())
        }
    }
}
