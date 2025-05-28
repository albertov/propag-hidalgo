use uom::si::f64::*;

#[derive(Clone, Copy, PartialEq)]
pub enum Life {
    Dead,
    Alive,
}

#[derive(Clone, Copy)]
pub enum ParticleType {
    Dead,       // Dead fuel particle
    Herb,       // Herbaceous live particle
    Wood,       // Woody live particle
    NoParticle, // Sentinel for no particle
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
    pub area_weight: f64,
    pub surface_area: f64,
    pub sigma_factor: f64,
    pub size_class_weight: f64,
    pub size_class: SizeClass,
    pub life: Life,
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

#[derive(Debug)]
pub struct Spread {
    pub rx_int: HeatFluxDensity,
    pub speed0: Velocity,
    pub hpua: RadiantExposure,
    pub phi_eff_wind: Ratio,
    pub speed_max: Velocity,
    pub azimuth_max: Angle,
    pub eccentricity: Ratio,
    pub residence_time: Time,
}

#[derive(Clone, Copy)]
pub struct Combustion {
    pub name: [u8; 16],
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
    pub sigma: f64,
    pub beta: f64,
    pub mext: f64,
    pub life_rx_factor_alive: f64,
    pub life_rx_factor_dead: f64,
    pub total_area: f64,
    pub alive_particles: Particles,
    pub dead_particles: Particles,
}

#[derive(PartialEq, Copy, Clone)]
pub enum SizeClass {
    SC0,
    SC1,
    SC2,
    SC3,
    SC4,
    SC5,
}

pub struct FuelDef {
    pub name: [u8; 16],
    pub desc: [u8; 64],
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub particles: ParticleDefs,
}

pub const MAX_PARTICLES: usize = 20;
pub const MAX_FUELS: usize = 20;

pub type ParticleDefs = [ParticleDef; MAX_PARTICLES];

#[inline]
pub fn iter_arr<const N: usize, T>(particles: &[Option<T>; N]) -> impl Iterator<Item = &T> {
    particles
        .iter()
        .take_while(|p| p.is_some())
        .map(|p| p.as_ref().unwrap())
}

#[derive(Clone, Copy)]
pub struct Fuel {
    pub name: [u8; 16],
    pub desc: [u8; 64],
    pub depth: f64,
    pub mext: f64,
    pub adjust: f64,
    pub alive_particles: Particles,
    pub dead_particles: Particles,
}

pub type Particles = [Particle; MAX_PARTICLES];

pub type Catalog = [Combustion; MAX_FUELS];

pub const fn init_arr<T: Copy, const N: usize, const M: usize>(def: T, src: [T; M]) -> [T; N] {
    let mut dst = [def; N];
    let mut i = 0;
    while i < M {
        dst[i] = src[i];
        i += 1;
    }
    dst
}

lazy_static::lazy_static! {
    pub static ref STANDARD_CATALOG : Catalog = {
        let mut dst = [Combustion::make(Fuel::SENTINEL); MAX_FUELS];
        let mut i = 0;
        while i < MAX_FUELS {
            dst[i] = Combustion::make(STANDARD_FUELS[i]);
            i += 1;
        }
        dst
    };
}

pub const STANDARD_FUELS: [Fuel; MAX_FUELS] = {
    init_arr(
        Fuel::SENTINEL,
        [
            Fuel::make(FuelDef::standard(
                *b"NoFuel",
                *b"No Combustible Fuel",
                0.1,
                0.01,
                [],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL01",
                *b"Short Grass (1 ft)",
                1.0,
                0.12,
                [ParticleDef::standard(ParticleType::Dead, 0.0340, 3500.0)],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL02",
                *b"Timber (grass & understory)",
                1.0,
                0.15,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.0920, 3000.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0460, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0230, 30.0),
                    ParticleDef::standard(ParticleType::Herb, 0.0230, 1500.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL03",
                *b"Tall Grass (2.5 ft)",
                2.5,
                0.25,
                [ParticleDef::standard(ParticleType::Dead, 0.1380, 1500.0)],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL04",
                *b"Chaparral (6 ft)",
                6.0,
                0.2,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.2300, 2000.0),
                    ParticleDef::standard(ParticleType::Dead, 0.1840, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0920, 30.0),
                    ParticleDef::standard(ParticleType::Wood, 0.2300, 1500.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL05",
                *b"Brush (2 ft)",
                2.0,
                0.2,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.0460, 2000.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0230, 109.0),
                    ParticleDef::standard(ParticleType::Wood, 0.0920, 1500.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL06",
                *b"Dormant Brush & Hardwood Slash",
                2.5,
                0.25,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.0690, 1750.0),
                    ParticleDef::standard(ParticleType::Dead, 0.1150, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0920, 30.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL07",
                *b"Southern Rough",
                2.5,
                0.40,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.0520, 1750.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0860, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0690, 30.0),
                    ParticleDef::standard(ParticleType::Wood, 0.0170, 1550.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL08",
                *b"Closed Timber Litter",
                0.2,
                0.30,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.0690, 2000.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0460, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.1150, 30.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL09",
                *b"Hardwood Litter",
                0.2,
                0.25,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.1340, 2500.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0190, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0070, 30.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL10",
                *b"Timber (litter & understory)",
                1.0,
                0.25,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.1380, 2000.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0920, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.2300, 30.0),
                    ParticleDef::standard(ParticleType::Wood, 0.0920, 1500.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL11",
                *b"Light Logging Slash",
                1.0,
                0.15,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.0690, 1500.0),
                    ParticleDef::standard(ParticleType::Dead, 0.2070, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.2530, 30.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL12",
                *b"Medium Logging Slash",
                2.3,
                0.20,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.1840, 1500.0),
                    ParticleDef::standard(ParticleType::Dead, 0.6440, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 0.7590, 30.0),
                ],
            )),
            Fuel::make(FuelDef::standard(
                *b"NFFL13",
                *b"Heavy Logging Slash",
                3.0,
                0.25,
                [
                    ParticleDef::standard(ParticleType::Dead, 0.3220, 1500.0),
                    ParticleDef::standard(ParticleType::Dead, 0.0580, 109.0),
                    ParticleDef::standard(ParticleType::Dead, 1.2880, 30.0),
                ],
            )),
        ],
    )
};
