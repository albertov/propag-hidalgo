use uom::si::f64::*;
use uom::si::length::foot;
use uom::si::ratio::ratio;

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
    pub savr: ReciprocalLength, // surface area to volume ratio
    pub density: MassDensity,
    pub heat: AvailableEnergy,
    pub si_total: Ratio,     // total silica content
    pub si_effective: Ratio, // effective silica content
}

#[derive(Debug, Clone, Copy)]
pub struct Terrain {
    pub d1hr: Ratio,
    pub d10hr: Ratio,
    pub d100hr: Ratio,
    pub herb: Ratio,
    pub wood: Ratio,
    pub wind_speed: Velocity,
    pub wind_azimuth: Angle,
    pub slope: Ratio,
    pub aspect: Angle,
}

pub struct Spread {
    pub rx_int: HeatFluxDensity,
    pub speed0: Velocity,
    pub hpua: RadiantExposure,
    pub phi_eff_wind: Ratio,
    pub speed_max: Velocity,
    pub azimuth_max: Angle,
    pub eccentricity: Ratio,
    pub byrams_max: LinearPowerDensity,
    pub flame_max: Length,
}

pub struct SpreadAtAzimuth {
    pub speed: Velocity,
    pub byrams: LinearPowerDensity,
    pub flame: Length,
}

pub struct Combustion {
    pub fuel: Fuel,
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

#[derive(Clone)]
pub struct Fuel {
    pub name: String,
    pub desc: String,
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub alive_particles: Vec<Particle>,
    pub dead_particles: Vec<Particle>,
}

pub type Catalog = Vec<Fuel>;

lazy_static::lazy_static! {
    pub static ref STANDARD_CATALOG : Catalog = {
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
    };
}
