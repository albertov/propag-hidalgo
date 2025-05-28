use crate::units::areal_mass_density::pound_per_square_foot;
use crate::units::heat_flux_density::btu_sq_foot_min;
use crate::units::linear_power_density::btu_foot_sec;
use crate::units::radiant_exposure::btu_sq_foot;
use uom::si::absement::foot_second;
use uom::si::angle::radian;
use uom::si::area::square_foot;
use uom::si::available_energy::btu_per_pound;
use uom::si::f64::*;
use uom::si::length::{foot, meter};
use uom::si::mass::kilogram;
use uom::si::mass_density::pound_per_cubic_foot;
use uom::si::ratio::ratio;
use uom::si::time::second;
use uom::si::velocity::foot_per_minute;

#[derive(Clone, Copy, PartialEq)]
pub enum Life {
    Dead,
    Alive,
}

#[derive(Clone, Copy)]
pub enum ParticleType {
    Dead, // Dead fuel particle
    Herb, // Herbaceous live particle
    Wood, // Woody live particle
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub type_: ParticleType,
    pub load: ArealMassDensity, // fuel loading
    pub savr: Length,           // surface area to volume ratio
    pub density: MassDensity,
    pub heat: AvailableEnergy,
    pub si_total: Ratio,     // total silica content
    pub si_effective: Ratio, // effective silica content
}

pub struct Terrain {
    pub d1hr: Ratio,
    pub d10hr: Ratio,
    pub d100hr: Ratio,
    pub herb: Ratio,
    pub wood: Ratio,
    pub wind_speed: Velocity,
    pub wind_azimuth: Angle,
    pub slope: Ratio,
    pub aspect: Ratio,
}

pub struct Spread {
    pub rx_int: HeatFluxDensity,
    pub speed: Velocity,
    pub hpua: RadiantExposure,
    pub phi_eff_wind: Ratio,
    pub speed_max: Velocity,
    pub azimuth_max: Angle,
    pub eccentricity: Ratio,
    pub byrams_max: LinearPowerDensity,
    pub flame_max: Length,
}

impl Spread {
    pub fn no_spread() -> Spread {
        Spread {
            rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(0.0),
            speed: Velocity::new::<foot_per_minute>(0.0),
            hpua: RadiantExposure::new::<btu_sq_foot>(0.0),
            phi_eff_wind: Ratio::new::<ratio>(0.0),
            speed_max: Velocity::new::<foot_per_minute>(0.0),
            azimuth_max: Angle::new::<radian>(0.0),
            eccentricity: Ratio::new::<ratio>(0.0),
            byrams_max: LinearPowerDensity::new::<btu_foot_sec>(0.0),
            flame_max: Length::new::<foot>(0.0),
        }
    }
}

pub struct Combustion<'a> {
    pub fuel: &'a Fuel,
    pub live_area_weight: Ratio,
    pub live_rx_factor: HeatFluxDensity,
    pub dead_area_weight: Ratio,
    pub dead_rx_factor: HeatFluxDensity,
    pub fine_dead_factor: ArealMassDensity,
    pub live_ext_factor: Ratio,
    pub fuel_bed_bulk_dens: MassDensity,
    pub residence_time: Time,
    pub flux_ratio: Ratio,
    pub slope_k: f64,
    pub wind_b: f64,
    pub wind_e: f64,
    pub wind_k: f64,
}

#[derive(PartialEq)]
pub enum SizeClass {
    SC0,
    SC1,
    SC2,
    SC3,
    SC4,
    SC5,
}

pub struct FuelDef {
    pub name: String,
    pub desc: String,
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub particles: Vec<Particle>,
}

pub struct Fuel {
    pub name: String,
    pub desc: String,
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub alive_particles: Vec<Particle>,
    pub dead_particles: Vec<Particle>,
}

impl Particle {
    pub fn standard(p_type: ParticleType, p_load: f64, p_savr: f64) -> Particle {
        Particle {
            type_: p_type,
            load: ArealMassDensity::new::<pound_per_square_foot>(p_load),
            savr: Length::new::<meter>(p_savr),
            density: MassDensity::new::<pound_per_cubic_foot>(32.0),
            heat: AvailableEnergy::new::<btu_per_pound>(8000.0),
            si_total: Ratio::new::<ratio>(0.0555),
            si_effective: Ratio::new::<ratio>(0.0100),
        }
    }
    pub const fn life(&self) -> Life {
        match &self.type_ {
            ParticleType::Dead => Life::Dead,
            _ => Life::Alive,
        }
    }

    pub fn moisture(&self, terrain: &Terrain) -> Ratio {
        match self.type_ {
            ParticleType::Herb => terrain.herb,
            ParticleType::Wood => terrain.wood,
            ParticleType::Dead => match self.size_class() {
                SizeClass::SC0 | SizeClass::SC1 => terrain.d1hr,
                SizeClass::SC2 | SizeClass::SC3 => terrain.d10hr,
                SizeClass::SC4 | SizeClass::SC5 => terrain.d100hr,
            },
        }
    }

    pub fn surface_area(&self) -> Area {
        if self.density.get::<pound_per_cubic_foot>() > SMIDGEN {
            self.load * self.savr / self.density
        } else {
            Area::new::<square_foot>(0.0)
        }
    }

    pub fn sigma_factor(&self) -> Ratio {
        (Length::new::<meter>(-138.0) / self.savr).exp()
    }

    pub fn size_class(&self) -> SizeClass {
        let savr = self.savr.get::<foot>();
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
            .partition(move |p| p.life() == Life::Alive);
        Fuel {
            name: def.name,
            desc: def.desc,
            depth: def.depth,
            mext: def.mext,
            adjust: def.adjust,
            alive_particles,
            dead_particles,
        }
    }
    fn particles(&self) -> impl Iterator<Item = &Particle> {
        self.alive_particles
            .iter()
            .chain(self.dead_particles.iter())
    }

    fn life_particles(&self, life: Life) -> impl Iterator<Item = &Particle> {
        match life {
            Life::Alive => self.alive_particles.iter(),
            Life::Dead => self.dead_particles.iter(),
        }
    }
    fn has_live_particles(&self) -> bool {
        self.life_particles(Life::Alive).count() > 0
    }
    fn total_area(&self) -> Area {
        self.particles().map(|p| p.surface_area()).sum()
    }
    fn life_area_weight(&self, life: Life) -> Ratio {
        self.life_particles(life)
            .map(|p| p.surface_area() / self.total_area())
            .sum()
    }
    fn life_fine_load(&self, life: Life) -> ArealMassDensity {
        match life {
            Life::Alive => self
                .life_particles(life)
                .map(|p| p.load * (Length::new::<meter>(-500.0) / p.savr).exp())
                .sum(),
            Life::Dead => self
                .life_particles(life)
                .map(|p| p.load * p.sigma_factor())
                .sum(),
        }
    }
    fn life_load(&self, life: Life) -> ArealMassDensity {
        self.life_particles(life)
            .map(|p| {
                self.part_size_class_weight(p) * p.load * (Ratio::new::<ratio>(1.0) - p.si_total)
            })
            .sum()
    }
    fn life_savr(&self, life: Life) -> Length {
        self.life_particles(life)
            .map(|p| self.part_area_weight(p) * p.savr)
            .sum()
    }
    fn life_heat(&self, life: Life) -> AvailableEnergy {
        self.life_particles(life)
            .map(|p| self.part_area_weight(p) * p.heat)
            .sum()
    }
    fn life_seff(&self, life: Life) -> Ratio {
        self.life_particles(life)
            .map(|p| self.part_area_weight(p) * p.si_effective)
            .sum()
    }
    fn life_eta_s(&self, life: Life) -> Ratio {
        let seff: f64 = self.life_seff(life).get::<ratio>();
        if seff > SMIDGEN {
            let eta = 0.174 / seff.powf(0.19);
            if eta < 1.0 {
                Ratio::new::<ratio>(eta)
            } else {
                Ratio::new::<ratio>(1.0)
            }
        } else {
            Ratio::new::<ratio>(0.0)
        }
    }
    fn sigma(&self) -> Length {
        self.life_area_weight(Life::Alive) * self.life_savr(Life::Alive)
            + self.life_area_weight(Life::Dead) * self.life_savr(Life::Dead)
    }
    fn ratio(&self) -> f64 {
        let sigma = self.sigma().get::<foot>();
        let beta = self.beta().get::<ratio>();
        beta / (3.348 / (sigma.powf(0.8189)))
    }
    fn flux_ratio(&self) -> Ratio {
        let sigma = self.sigma().get::<foot>();
        let beta = self.beta().get::<ratio>();
        Ratio::new::<ratio>(
            ((0.792 + 0.681 * sigma.sqrt()) * (beta.sqrt() + 0.1)).exp() / (192.0 + 0.2595 * sigma),
        )
    }
    fn beta(&self) -> Ratio {
        if self.depth.get::<foot>() > SMIDGEN {
            let x: Length = self.particles().map(|p| p.load / p.density).sum();
            x / self.depth
        } else {
            Ratio::new::<ratio>(0.0)
        }
    }
    fn gamma(&self) -> Ratio {
        let sigma = self.sigma().get::<foot>();
        let sigma15 = sigma.powf(1.5);
        let gamma_max = sigma15 / (495.0 + 0.0594 * sigma15);
        let aa = 133.0 / sigma.powf(0.7913);
        let r = gamma_max * self.ratio().powf(aa) * (aa * (1.0 - self.ratio())).exp();
        Ratio::new::<ratio>(r)
    }

    fn life_rx_factor(&self, life: Life) -> HeatFluxDensity {
        self.life_load(life) * self.life_heat(life) * self.life_eta_s(life) * self.gamma()
            / Time::new::<second>(1.0)
    }
    fn part_area_weight(&self, particle: &Particle) -> Ratio {
        let total_area_life: Area = self
            .life_particles(particle.life())
            .map(|p| p.surface_area())
            .sum();
        if total_area_life.get::<square_foot>() > SMIDGEN {
            particle.surface_area() / total_area_life
        } else {
            Ratio::new::<ratio>(0.0)
        }
    }

    fn part_size_class_weight(&self, particle: &Particle) -> Ratio {
        self.life_particles(particle.life())
            .filter(|p| p.size_class() == particle.size_class())
            .map(|p| self.part_area_weight(p))
            .sum()
    }

    fn live_ext_factor(&self) -> Ratio {
        Ratio::new::<ratio>(2.9) * self.life_fine_load(Life::Dead)
            / self.life_fine_load(Life::Alive)
    }

    fn bulk_density(&self) -> MassDensity {
        self.particles().map(|p| p.load / self.depth).sum()
    }

    fn residence_time(&self) -> Time {
        Absement::new::<foot_second>(384.0 / 60.0) / self.sigma()
    }

    fn slope_k(&self) -> f64 {
        let beta = self.beta().get::<ratio>();
        5.275 * beta.powf(-0.3)
    }

    fn wind_bke(&self) -> (f64, f64, f64) {
        let sigma = self.sigma().get::<foot>();
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
            fuel: self,
            live_area_weight: self.life_area_weight(Life::Alive),
            live_rx_factor: self.life_rx_factor(Life::Alive),
            dead_area_weight: self.life_area_weight(Life::Dead),
            dead_rx_factor: self.life_rx_factor(Life::Dead),
            fine_dead_factor: self.life_load(Life::Dead),
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
const SMIDGEN: f64 = 1e-6;

pub type Catalog = Vec<Fuel>;

pub fn standard_catalog() -> Catalog {
    vec![
        Fuel::make(FuelDef {
            name: String::from("NoFuel"),
            desc: String::from("No Combustible Fuel"),
            depth: Length::new::<foot>(0.1),
            mext: Ratio::new::<ratio>(0.01),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![],
        }),
        Fuel::make(FuelDef {
            name: String::from("NFFL01"),
            desc: String::from("Short Grass (1 ft)"),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.12),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![Particle::standard(ParticleType::Dead, 0.0340, 3500.0)],
        }),
        Fuel::make(FuelDef {
            name: String::from("NFFL02"),
            desc: String::from("Timber (grass & understory)"),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.15),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                Particle::standard(ParticleType::Dead, 0.0920, 3000.0),
                Particle::standard(ParticleType::Dead, 0.0460, 109.0),
                Particle::standard(ParticleType::Dead, 0.0230, 30.0),
                Particle::standard(ParticleType::Herb, 0.0230, 1500.0),
            ],
        }),
        Fuel::make(FuelDef {
            name: String::from("NFFL03"),
            desc: String::from("Tall Grass (2.5 ft)"),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![Particle::standard(ParticleType::Dead, 0.1380, 1500.0)],
        }),
        Fuel::make(FuelDef {
            name: String::from("NFFL04"),
            desc: String::from("Chaparral (6 ft)"),
            depth: Length::new::<foot>(6.0),
            mext: Ratio::new::<ratio>(0.2),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                Particle::standard(ParticleType::Dead, 0.2300, 2000.0),
                Particle::standard(ParticleType::Dead, 0.1840, 109.0),
                Particle::standard(ParticleType::Dead, 0.0920, 30.0),
                Particle::standard(ParticleType::Wood, 0.2300, 1500.0),
            ],
        }),
        Fuel::make(FuelDef {
            name: String::from("NFFL05"),
            desc: String::from("Brush (2 ft)"),
            depth: Length::new::<foot>(2.0),
            mext: Ratio::new::<ratio>(0.2),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                Particle::standard(ParticleType::Dead, 0.0460, 2000.0),
                Particle::standard(ParticleType::Dead, 0.0230, 109.0),
                Particle::standard(ParticleType::Wood, 0.0920, 1500.0),
            ],
        }),
        Fuel::make(FuelDef {
            name: String::from("NFFL06"),
            desc: String::from("Dormant Brush & Hardwood Slash"),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                Particle::standard(ParticleType::Dead, 0.0690, 1750.0),
                Particle::standard(ParticleType::Dead, 0.1150, 109.0),
                Particle::standard(ParticleType::Wood, 0.0920, 30.0),
            ],
        }),
    ]
}

impl Combustion<'_> {
    pub fn spread(&self, terrain: &Terrain) -> Spread {
        if self.fuel.alive_particles.is_empty() && self.fuel.dead_particles.is_empty() {
            Spread::no_spread()
        } else {
            Spread {
                rx_int: self.rx_int(terrain),
                speed: self.speed(terrain),
                hpua: self.hpua(terrain),
                phi_eff_wind: self.phi_eff_wind(terrain),
                speed_max: self.speed_max(terrain),
                azimuth_max: self.azimuth_max(terrain),
                eccentricity: self.eccentricity(terrain),
                byrams_max: self.byrams_max(terrain),
                flame_max: self.flame_max(terrain),
            }
        }
    }
    fn rx_int(&self, terrain: &Terrain) -> HeatFluxDensity {
        self.fuel.life_rx_factor(Life::Alive) * self.life_eta_m(Life::Alive, terrain)
            + self.fuel.life_rx_factor(Life::Dead) * self.life_eta_m(Life::Dead, terrain)
    }
    fn speed(&self, terrain: &Terrain) -> Velocity {
        (self.rx_int(terrain) * self.flux_ratio / self.rbqig(terrain))
            * Length::new::<meter>(1.0)
            * (Time::new::<second>(1.0) * Time::new::<second>(1.0))
            / Mass::new::<kilogram>(1.0)
    }
    fn hpua(&self, terrain: &Terrain) -> RadiantExposure {
        self.rx_int(terrain) * self.residence_time
    }
    fn phi_eff_wind(&self, terrain: &Terrain) -> Ratio {
        todo!()
    }
    fn speed_max(&self, terrain: &Terrain) -> Velocity {
        self.speed(terrain) * (Ratio::new::<ratio>(1.0) + self.phi_ew(terrain))
    }
    fn azimuth_max(&self, terrain: &Terrain) -> Angle {
        todo!()
    }
    fn eccentricity(&self, terrain: &Terrain) -> Ratio {
        todo!()
    }
    fn byrams_max(&self, terrain: &Terrain) -> LinearPowerDensity {
        self.residence_time * self.speed_max(terrain) * self.rx_int(terrain)
            / Ratio::new::<ratio>(60.0)
    }
    fn flame_max(&self, terrain: &Terrain) -> Length {
        flame_length(self.byrams_max(terrain))
    }

    fn life_moisture(&self, life: Life, terrain: &Terrain) -> Ratio {
        self.fuel
            .life_particles(life)
            .map(|p| self.fuel.part_area_weight(p) * p.moisture(terrain))
            .sum()
    }

    fn life_mext(&self, life: Life, terrain: &Terrain) -> Ratio {
        match life {
            Life::Alive => {
                if self.fuel.has_live_particles() {
                    let fdmois = self.wfmd(terrain) / self.fine_dead_factor;
                    let live_mext = self.live_ext_factor
                        * (Ratio::new::<ratio>(1.0) - fdmois / self.fuel.mext)
                        - Ratio::new::<ratio>(0.226);
                    live_mext.max(self.fuel.mext)
                } else {
                    Ratio::new::<ratio>(0.0)
                }
            }
            Life::Dead => self.fuel.mext,
        }
    }

    fn life_eta_m(&self, life: Life, terrain: &Terrain) -> Ratio {
        if self.life_moisture(life, terrain) >= self.life_mext(life, terrain) {
            Ratio::new::<ratio>(0.0)
        } else if self.life_mext(life, terrain).get::<ratio>() > SMIDGEN {
            let rt =
                (self.life_moisture(life, terrain) / self.life_mext(life, terrain)).get::<ratio>();
            let x = 1.0 - 2.59 * rt.powf(1.0) + 5.11 * rt.powf(2.0) - 3.52 * rt.powf(3.0);
            Ratio::new::<ratio>(x)
        } else {
            Ratio::new::<ratio>(0.0)
        }
    }

    fn wfmd(&self, terrain: &Terrain) -> ArealMassDensity {
        self.fuel
            .life_particles(Life::Dead)
            .map(|p| p.moisture(terrain) * p.sigma_factor() * p.load)
            .sum()
    }

    fn rbqig(&self, terrain: &Terrain) -> Ratio {
        self.fuel
            .particles()
            .map(|p| {
                (Ratio::new::<ratio>(250.0) + Ratio::new::<ratio>(1116.0) * p.moisture(terrain))
                    * self.fuel.part_area_weight(p)
                    * self.fuel.life_area_weight(p.life())
                    * p.sigma_factor()
            })
            .sum()
    }

    fn phi_slope(&self, terrain: &Terrain) -> Ratio {
        Ratio::new::<ratio>(self.slope_k * terrain.slope.get::<ratio>().powf(2.0))
    }

    fn phi_wind(&self, terrain: &Terrain) -> Ratio {
        if terrain.wind_speed.get::<foot_per_minute>() > SMIDGEN {
            let ws = terrain.wind_speed.get::<foot_per_minute>();
            Ratio::new::<ratio>(self.wind_k * ws.powf(self.wind_b))
        } else {
            Ratio::new::<ratio>(0.0)
        }
    }

    fn phi_ew(&self, terrain: &Terrain) -> Ratio {
        self.phi_slope(terrain) + self.phi_wind(terrain)
    }
}

fn flame_length(byrams: LinearPowerDensity) -> Length {
    let b = byrams.get::<btu_foot_sec>();
    if b > SMIDGEN {
        Length::new::<foot>(0.45 * b.powf(0.46))
    } else {
        Length::new::<foot>(0.0)
    }
}
