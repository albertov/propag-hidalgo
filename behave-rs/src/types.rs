use heapless::String;
use heapless::Vec;
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
pub struct ParticleDef {
    pub type_: ParticleType,
    pub load: ArealMassDensity, // fuel loading
    pub savr: ReciprocalLength, // surface area to volume ratio
    pub density: MassDensity,
    pub heat: AvailableEnergy,
    pub si_total: Ratio,     // total silica content
    pub si_effective: Ratio, // effective silica content
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub type_: ParticleType,
    pub load: f64,
    pub savr: f64,
    pub density: f64,
    pub heat: f64,
    pub si_total: f64,
    pub si_effective: f64,
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
    pub live_area_weight: f64,
    pub live_rx_factor: f64,
    pub dead_area_weight: f64,
    pub dead_rx_factor: f64,
    pub fine_dead_factor: f64,
    pub live_ext_factor: f64,
    pub fuel_bed_bulk_dens: f64,
    pub residence_time: f64,
    pub flux_ratio: f64,
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
    pub name: String<16>,
    pub desc: String<64>,
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub particles: Vec<ParticleDef, 20>,
}

#[derive(Clone)]
pub struct Fuel {
    pub name: String<16>,
    pub desc: String<64>,
    pub depth: f64,
    pub mext: f64,
    pub adjust: f64,
    pub alive_particles: Vec<Particle, 20>,
    pub dead_particles: Vec<Particle, 20>,
}

pub type Catalog = Vec<Fuel, 20>;

lazy_static::lazy_static! {
pub static ref STANDARD_CATALOG : Catalog = {
    Vec::from_slice(&[
        Fuel::make(FuelDef {
            name: String::try_from("NoFuel").unwrap(),
            desc: String::try_from("No Combustible Fuel").unwrap(),
            depth: Length::new::<foot>(0.1),
            mext: Ratio::new::<ratio>(0.01),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::new(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL01").unwrap(),
            desc: String::try_from("Short Grass (1 ft)").unwrap(),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.12),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[ParticleDef::standard(ParticleType::Dead, 0.0340, 3500.0)]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL02").unwrap(),
            desc: String::try_from("Timber (grass & understory)").unwrap(),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.15),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.0920, 3000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0460, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0230, 30.0),
                ParticleDef::standard(ParticleType::Herb, 0.0230, 1500.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL03").unwrap(),
            desc: String::try_from("Tall Grass (2.5 ft)").unwrap(),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[ParticleDef::standard(ParticleType::Dead, 0.1380, 1500.0)]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL04").unwrap(),
            desc: String::try_from("Chaparral (6 ft)").unwrap(),
            depth: Length::new::<foot>(6.0),
            mext: Ratio::new::<ratio>(0.2),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.2300, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.1840, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0920, 30.0),
                ParticleDef::standard(ParticleType::Wood, 0.2300, 1500.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL05").unwrap(),
            desc: String::try_from("Brush (2 ft)").unwrap(),
            depth: Length::new::<foot>(2.0),
            mext: Ratio::new::<ratio>(0.2),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.0460, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0230, 109.0),
                ParticleDef::standard(ParticleType::Wood, 0.0920, 1500.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL06").unwrap(),
            desc: String::try_from("Dormant Brush & Hardwood Slash").unwrap(),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.0690, 1750.0),
                ParticleDef::standard(ParticleType::Dead, 0.1150, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0920, 30.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL07").unwrap(),
            desc: String::try_from("Southern Rough").unwrap(),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.40),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.0520, 1750.0),
                ParticleDef::standard(ParticleType::Dead, 0.0860, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0690, 30.0),
                ParticleDef::standard(ParticleType::Wood, 0.0170, 1550.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL08").unwrap(),
            desc: String::try_from("Closed Timber Litter").unwrap(),
            depth: Length::new::<foot>(0.2),
            mext: Ratio::new::<ratio>(0.30),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.0690, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0460, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.1150, 30.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL09").unwrap(),
            desc: String::try_from("Hardwood Litter").unwrap(),
            depth: Length::new::<foot>(0.2),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.1340, 2500.0),
                ParticleDef::standard(ParticleType::Dead, 0.0190, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.0070, 30.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL10").unwrap(),
            desc: String::try_from("Timber (litter & understory)").unwrap(),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.1380, 2000.0),
                ParticleDef::standard(ParticleType::Dead, 0.0920, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.2300, 30.0),
                ParticleDef::standard(ParticleType::Wood, 0.0920, 1500.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL11").unwrap(),
            desc: String::try_from("Light Logging Slash").unwrap(),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.15),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.0690, 1500.0),
                ParticleDef::standard(ParticleType::Dead, 0.2070, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.2530, 30.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL12").unwrap(),
            desc: String::try_from("Medium Logging Slash").unwrap(),
            depth: Length::new::<foot>(2.3),
            mext: Ratio::new::<ratio>(0.20),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.1840, 1500.0),
                ParticleDef::standard(ParticleType::Dead, 0.6440, 109.0),
                ParticleDef::standard(ParticleType::Dead, 0.7590, 30.0),
            ]).unwrap(),
        }),
        Fuel::make(FuelDef {
            name: String::try_from("NFFL13").unwrap(),
            desc: String::try_from("Heavy Logging Slash").unwrap(),
            depth: Length::new::<foot>(3.0),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: Vec::from_slice(&[
                ParticleDef::standard(ParticleType::Dead, 0.3220, 1500.0),
                ParticleDef::standard(ParticleType::Dead, 0.0580, 109.0),
                ParticleDef::standard(ParticleType::Dead, 1.2880, 30.0),
            ]).unwrap(),
        }),
    ]).unwrap()
    };
}
