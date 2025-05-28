use crate::units::areal_mass_density::pound_per_square_foot;
use crate::units::reciprocal_length::per_foot;
use uom::si::available_energy::btu_per_pound;
use uom::si::f64::*;
use uom::si::length::foot;
use uom::si::mass_density::pound_per_cubic_foot;
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
    pub part_type: ParticleType,
    pub part_load: ArealMassDensity, // fuel loading
    pub part_savr: ReciprocalLength, // surface area to volume ratio
    pub part_density: MassDensity,
    pub part_heat: AvailableEnergy,
    pub part_si_total: Ratio,     // total silica content
    pub part_si_effective: Ratio, // effective silica content
}

impl Particle {
    pub fn part_life(&self) -> Life {
        match &self.part_type {
            ParticleType::Dead => Life::Dead,
            _ => Life::Alive,
        }
    }
}

pub enum SizeClass {
    SC0,
    SC1,
    SC2,
    SC3,
    SC4,
    SC5,
}

pub struct Fuel {
    pub name: String,
    pub desc: String,
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub particles: Vec<Particle>,
}

impl Fuel {
    pub fn life_particles(&self, life: Life) -> impl Iterator<Item = &Particle> {
        self.particles
            .iter()
            .filter(move |&p| p.part_life() == life)
    }
}

pub type Catalog = Vec<Fuel>;

fn mk_part(p_type: ParticleType, p_load: f64, p_savr: f64) -> Particle {
    Particle {
        part_type: p_type,
        part_load: ArealMassDensity::new::<pound_per_square_foot>(p_load),
        part_savr: ReciprocalLength::new::<per_foot>(p_savr),
        part_density: MassDensity::new::<pound_per_cubic_foot>(32.0),
        part_heat: AvailableEnergy::new::<btu_per_pound>(8000.0),
        part_si_total: Ratio::new::<ratio>(0.0555),
        part_si_effective: Ratio::new::<ratio>(0.0100),
    }
}

pub fn standard_catalog() -> Catalog {
    vec![
        Fuel {
            name: String::from("NoFuel"),
            desc: String::from("No Combustible Fuel"),
            depth: Length::new::<foot>(0.1),
            mext: Ratio::new::<ratio>(0.01),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![],
        },
        Fuel {
            name: String::from("NFFL01"),
            desc: String::from("Short Grass (1 ft)"),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.12),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![mk_part(ParticleType::Dead, 0.0340, 3500.0)],
        },
        Fuel {
            name: String::from("NFFL02"),
            desc: String::from("Timber (grass & understory)"),
            depth: Length::new::<foot>(1.0),
            mext: Ratio::new::<ratio>(0.15),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                mk_part(ParticleType::Dead, 0.0920, 3000.0),
                mk_part(ParticleType::Dead, 0.0460, 109.0),
                mk_part(ParticleType::Dead, 0.0230, 30.0),
                mk_part(ParticleType::Herb, 0.0230, 1500.0),
            ],
        },
        Fuel {
            name: String::from("NFFL03"),
            desc: String::from("Tall Grass (2.5 ft)"),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![mk_part(ParticleType::Dead, 0.1380, 1500.0)],
        },
        Fuel {
            name: String::from("NFFL04"),
            desc: String::from("Chaparral (6 ft)"),
            depth: Length::new::<foot>(6.0),
            mext: Ratio::new::<ratio>(0.2),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                mk_part(ParticleType::Dead, 0.2300, 2000.0),
                mk_part(ParticleType::Dead, 0.1840, 109.0),
                mk_part(ParticleType::Dead, 0.0920, 30.0),
                mk_part(ParticleType::Wood, 0.2300, 1500.0),
            ],
        },
        Fuel {
            name: String::from("NFFL05"),
            desc: String::from("Brush (2 ft)"),
            depth: Length::new::<foot>(2.0),
            mext: Ratio::new::<ratio>(0.2),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                mk_part(ParticleType::Dead, 0.0460, 2000.0),
                mk_part(ParticleType::Dead, 0.0230, 109.0),
                mk_part(ParticleType::Wood, 0.0920, 1500.0),
            ],
        },
        Fuel {
            name: String::from("NFFL06"),
            desc: String::from("Dormant Brush & Hardwood Slash"),
            depth: Length::new::<foot>(2.5),
            mext: Ratio::new::<ratio>(0.25),
            adjust: Ratio::new::<ratio>(1.0),
            particles: vec![
                mk_part(ParticleType::Dead, 0.0690, 1750.0),
                mk_part(ParticleType::Dead, 0.1150, 109.0),
                mk_part(ParticleType::Wood, 0.0920, 30.0),
            ],
        },
    ]
}
