use crate::units::areal_mass_density::pound_per_square_foot;
use uom::si::area::square_meter;
use uom::si::available_energy::btu_per_pound;
use uom::si::f64::*;
use uom::si::length::{foot, meter};
use uom::si::mass::pound;
use uom::si::mass_density::pound_per_cubic_foot;
use uom::si::ratio::ratio;
use uom::si::time::{minute, second};

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

impl Particle {
    pub fn life(&self) -> Life {
        match &self.type_ {
            ParticleType::Dead => Life::Dead,
            _ => Life::Alive,
        }
    }

    pub fn surface_area(&self) -> Area {
        if self.density.get::<pound_per_cubic_foot>() > SMIDGEN {
            self.load * self.savr / self.density
        } else {
            Area::new::<square_meter>(0.0)
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

#[derive(PartialEq)]
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

pub struct Combustion<'a> {
    pub fuel: &'a Fuel,
    pub live_area_weight: Ratio,
    pub live_rx_factor: Ratio,
    pub dead_area_weight: Ratio,
    pub dead_rx_factor: Ratio,
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

impl Fuel {
    pub fn life_particles(&self, life: Life) -> impl Iterator<Item = &Particle> {
        self.particles.iter().filter(move |&p| p.life() == life)
    }
    fn total_area(&self) -> Area {
        self.particles.iter().map(|p| p.surface_area()).sum()
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
    fn life_eta(&self, life: Life) -> Ratio {
        let seff: f64 = self.life_seff(life).get::<ratio>();
        if seff <= SMIDGEN {
            Ratio::new::<ratio>(0.0)
        } else {
            let eta = 0.174 / seff.powf(0.19);
            if eta < 1.0 {
                Ratio::new::<ratio>(eta)
            } else {
                Ratio::new::<ratio>(1.0)
            }
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
    fn beta(&self) -> Ratio {
        if self.depth.get::<foot>() > SMIDGEN {
            let x: Length = self.particles.iter().map(|p| p.load / p.density).sum();
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

    fn life_rx_factor(&self, life: Life) -> Ratio {
        self.life_load(life)
            * self.life_heat(life)
            * self.life_eta(life)
            * self.gamma()
            * (Time::new::<second>(1.0) * Time::new::<second>(1.0))
            / Mass::new::<pound>(1.0)
    }
    fn part_area_weight(&self, particle: &Particle) -> Ratio {
        let total_area_life: Area = self
            .life_particles(particle.life())
            .map(|p| p.surface_area())
            .sum();
        if total_area_life.get::<square_meter>() > SMIDGEN {
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

    pub fn combustion(&self) -> Combustion {
        //TBD
        Combustion {
            fuel: self,
            live_area_weight: self.life_area_weight(Life::Alive),
            live_rx_factor: self.life_rx_factor(Life::Alive),
            dead_area_weight: self.life_area_weight(Life::Dead),
            dead_rx_factor: self.life_rx_factor(Life::Dead),
            fine_dead_factor: self.life_load(Life::Dead),
            live_ext_factor: self.live_ext_factor(),
            fuel_bed_bulk_dens: MassDensity::new::<pound_per_cubic_foot>(1.0),
            residence_time: Time::new::<minute>(1.0),
            flux_ratio: Ratio::new::<ratio>(1.0),
            slope_k: 1.0,
            wind_b: 1.0,
            wind_e: 1.0,
            wind_k: 1.0,
        }
    }
}
const SMIDGEN: f64 = 1e-6;

pub type Catalog = Vec<Fuel>;

fn mk_part(p_type: ParticleType, p_load: f64, p_savr: f64) -> Particle {
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
