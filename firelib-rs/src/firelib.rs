#[cfg(feature = "std")]
extern crate std;

#[cfg(target_os = "cuda")]
extern crate cuda_std;

#[cfg(not(target_os = "cuda"))]
extern crate cust;

use crate::units::heat_flux_density::btu_sq_foot_min;
use crate::units::linear_power_density::btu_foot_sec;
use crate::units::radiant_exposure::btu_sq_foot;
use crate::units::*;
use const_soft_float::soft_f64::SoftF64;
use uom::si::angle::radian;
use uom::si::f64::*;
use uom::si::length::foot;
use uom::si::ratio::ratio;
use uom::si::time::minute;
use uom::si::velocity::foot_per_minute;
use uom::si::velocity::meter_per_second;

pub(crate) const SMIDGEN: f64 = 1e-6;
const PI: f64 = 3.141592653589793;
const MAX_PARTICLES: usize = 5;
const MAX_FUELS: usize = 20;

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

#[cfg_attr(not(target_os = "cuda"), derive(Copy, Clone, Debug, cust::DeviceCopy))]
pub struct TerrainCuda {
    pub d1hr: f64,
    pub d10hr: f64,
    pub d100hr: f64,
    pub herb: f64,
    pub wood: f64,
    pub wind_speed: f64,
    pub wind_azimuth: f64,
    pub slope: f64,
    pub aspect: f64,
}

macro_rules! to_quantity {
    ($quant:ident, $val:expr) => {
	{
	    use uom::lib::marker::PhantomData;
	    $quant { value: $val, units: PhantomData, dimension: PhantomData}
	}
    };
}
macro_rules! from_quantity {
    ($quant:ident, $val:expr) => {
	{
	    let $quant { value, .. } = $val;
	    value
	}
    };
}
impl From<TerrainCuda> for Terrain {
    fn from(f: TerrainCuda) -> Self {
        Self {
            d1hr: to_quantity!(Ratio, f.d1hr),
            d10hr: to_quantity!(Ratio, f.d10hr),
            d100hr: to_quantity!(Ratio, f.d100hr),
            herb: to_quantity!(Ratio, f.herb),
            wood: to_quantity!(Ratio, f.wood),
            wind_speed: to_quantity!(Velocity, f.wind_speed),
            wind_azimuth: to_quantity!(Angle, f.wind_azimuth),
            slope: to_quantity!(Ratio, f.slope),
            aspect: to_quantity!(Angle, f.aspect),
        }
    }
}
impl From<Terrain> for TerrainCuda {
    fn from(f: Terrain) -> Self {
        Self {
            d1hr: from_quantity!(Ratio, f.d1hr),
            d10hr: from_quantity!(Ratio, f.d10hr),
            d100hr: from_quantity!(Ratio, f.d100hr),
            herb: from_quantity!(Ratio, f.herb),
            wood: from_quantity!(Ratio, f.wood),
            wind_speed: from_quantity!(Velocity, f.wind_speed),
            wind_azimuth: from_quantity!(Angle, f.wind_azimuth),
            slope: from_quantity!(Ratio, f.slope),
            aspect: from_quantity!(Angle, f.aspect),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fire {
    pub rx_int: HeatFluxDensity,
    pub speed0: Velocity,
    pub hpua: RadiantExposure,
    pub phi_eff_wind: Ratio,
    pub speed_max: Velocity,
    pub azimuth_max: Angle,
    pub eccentricity: Ratio,
    pub residence_time: Time,
}
#[cfg_attr(not(target_os = "cuda"), derive(Copy, Clone, Debug, cust::DeviceCopy))]
pub struct FireCuda {
    pub rx_int: f64,
    pub speed0: f64,
    pub hpua: f64,
    pub phi_eff_wind: f64,
    pub speed_max: f64,
    pub azimuth_max: f64,
    pub eccentricity: f64,
    pub residence_time: f64,
}


impl From<FireCuda> for Fire {
    fn from(f: FireCuda) -> Self {
        Self {
            rx_int: to_quantity!(HeatFluxDensity, f.rx_int),
            speed0: to_quantity!(Velocity, f.speed0),
            hpua: to_quantity!(RadiantExposure, f.hpua),
            phi_eff_wind: to_quantity!(Ratio, f.phi_eff_wind),
            speed_max: to_quantity!(Velocity, f.speed_max),
            azimuth_max: to_quantity!(Angle, f.azimuth_max),
            eccentricity: to_quantity!(Ratio, f.eccentricity),
            residence_time: to_quantity!(Time, f.residence_time),
        }
    }
}
impl From<Fire> for FireCuda {
    fn from(f: Fire) -> Self {
        Self {
            rx_int: from_quantity!(HeatFluxDensity, f.rx_int),
            speed0: from_quantity!(Velocity, f.speed0),
            hpua: from_quantity!(RadiantExposure, f.hpua),
            phi_eff_wind: from_quantity!(Ratio, f.phi_eff_wind),
            speed_max: from_quantity!(Velocity, f.speed_max),
            azimuth_max: from_quantity!(Angle, f.azimuth_max),
            eccentricity: from_quantity!(Ratio, f.eccentricity),
            residence_time: from_quantity!(Time, f.residence_time),
        }
    }
}

#[derive(Debug)]
pub struct Spread<'a> {
    fire: &'a Fire,
    factor: Ratio,
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
    pub alive_particles: Particles,
    pub dead_particles: Particles,
}

pub type ParticleDefs = [ParticleDef; MAX_PARTICLES];

#[derive(Clone, Copy)]
pub struct Fuel {
    pub name: [u8; 16],
    pub desc: [u8; 64],
    pub depth: f64,
    pub mext: f64,
    pub adjust: f64,
    pub alive_particles: Particles,
    pub dead_particles: Particles,
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
    pub life_rx_factor_alive: f64,
    pub life_rx_factor_dead: f64,
    pub total_area: f64,
}

pub type Particles = [Particle; MAX_PARTICLES];

pub struct Catalog([Fuel; MAX_FUELS]);

macro_rules! accum_particles {
    ($particles:expr, $fun:expr) => ({
        match(&$particles, &$fun) {
            (particles, fun) => {
                let mut r = 0.0;
                let mut i = 0;
                while i < particles.len() {
                    let p = &particles[i];
                    if p.is_sentinel() {
                        break;
                    }
                    r += fun(p);
                    i += 1
                }
                r
            }
        }
    });
    ($particles:expr, $fun:expr, $( $args:expr ),*) => ({
        match(&$particles, &$fun) {
            (particles, fun) => {
                let mut r = 0.0;
                let mut i = 0;
                while i < particles.len() {
                    let p = &particles[i];
                    if p.is_sentinel() {
                        break;
                    }
                    r += fun(p, $( $args ),*);
                    i += 1
                }
                r
            }
        }
    });
}

impl Catalog {
    pub const fn make<const N: usize>(fuels: [Fuel; N]) -> Self {
        Self(init_arr(Fuel::SENTINEL, fuels))
    }
    pub fn get(&self, idx: usize) -> Option<&Fuel> {
        match self.0.get(idx).as_ref() {
            Some(x) if x.has_particles() => Some(x),
            _ => None,
        }
    }
}

impl Terrain {
    fn upslope(&self) -> f64 {
        let aspect = self.aspect.get::<radian>();
        if aspect >= PI {
            aspect - PI
        } else {
            aspect + PI
        }
    }
}

impl<'a> Fire {
    pub fn null() -> Self {
        Self {
            rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(0.0),
            speed0: Velocity::new::<foot_per_minute>(0.0),
            hpua: RadiantExposure::new::<btu_sq_foot>(0.0),
            phi_eff_wind: Ratio::new::<ratio>(0.0),
            speed_max: Velocity::new::<foot_per_minute>(0.0),
            azimuth_max: Angle::new::<radian>(0.0),
            eccentricity: Ratio::new::<ratio>(0.0),
            residence_time: Time::new::<minute>(0.0),
        }
    }
    fn flame_length(byrams: f64) -> f64 {
        if byrams > SMIDGEN {
            0.45 * f64::powf(byrams, 0.46)
        } else {
            0.0
        }
    }
    pub fn byrams_max(&self) -> LinearPowerDensity {
        LinearPowerDensity::new::<btu_foot_sec>(
            self.residence_time.get::<minute>()
                * self.speed_max.get::<foot_per_minute>()
                * self.rx_int.get::<btu_sq_foot_min>()
                / 60.0,
        )
    }
    pub fn flame_max(&self) -> Length {
        Length::new::<foot>(Self::flame_length(self.byrams_max().get::<btu_foot_sec>()))
    }

    pub fn spread(&'a self, azimuth: Angle) -> Spread<'a> {
        let azimuth = azimuth.get::<radian>();
        let azimuth_max = self.azimuth_max.get::<radian>();
        let angle = {
            let ret = (azimuth - azimuth_max).abs();
            if ret > PI {
                2.0 * PI - ret
            } else {
                ret
            }
        };
        let ecc = self.eccentricity.get::<ratio>();
        let factor = if (azimuth - azimuth_max).abs() < SMIDGEN {
            1.0
        } else {
            safe_div(1.0 - ecc, 1.0 - ecc * f64::cos(angle))
        };
        Spread {
            fire: self,
            factor: Ratio::new::<ratio>(factor),
        }
    }

    pub fn almost_eq(&self, other: &Self) -> bool {
        #[allow(unused)]
        fn cmp(msg: &str, a: f64, b: f64) -> bool {
            let r = (a - b).abs() < SMIDGEN;
            #[cfg(feature = "std")]
            #[cfg(test)]
            if !r {
                std::println!("{}: {} /= {}", msg, a, b);
            }
            r
        }
        cmp(
            "rx_int",
            self.rx_int.get::<btu_sq_foot_min>(),
            other.rx_int.get::<btu_sq_foot_min>(),
        ) && cmp(
            "speed0",
            self.speed0.get::<foot_per_minute>(),
            other.speed0.get::<foot_per_minute>(),
        ) && cmp(
            "hpua",
            self.hpua.get::<btu_sq_foot>(),
            other.hpua.get::<btu_sq_foot>(),
        ) && cmp(
            "phi_eff_wind",
            self.phi_eff_wind.get::<ratio>(),
            other.phi_eff_wind.get::<ratio>(),
        ) && cmp(
            "speed_max",
            self.speed_max.get::<foot_per_minute>(),
            other.speed_max.get::<foot_per_minute>(),
        ) && cmp(
            "eccentricity",
            self.eccentricity.get::<ratio>(),
            other.eccentricity.get::<ratio>(),
        ) && cmp(
            "byrams_max",
            self.byrams_max().get::<btu_foot_sec>(),
            other.byrams_max().get::<btu_foot_sec>(),
        ) && cmp(
            "flame_max",
            self.flame_max().get::<foot>(),
            other.flame_max().get::<foot>(),
        ) && cmp(
            "azimuth_max",
            self.azimuth_max.get::<radian>(),
            other.azimuth_max.get::<radian>(),
        )
    }
}

impl<'a> Spread<'a> {
    pub fn speed(&self) -> Velocity {
        self.fire.speed_max * self.factor
    }
    pub fn byrams(&self) -> LinearPowerDensity {
        self.fire.byrams_max() * self.factor
    }
    pub fn flame(&self) -> Length {
        Length::new::<foot>(Fire::flame_length(self.byrams().get::<btu_foot_sec>()))
    }
}

impl ParticleDef {
    const SENTINEL: Self = ParticleDef::standard(ParticleType::NoParticle, 0.0, 0.0);

    pub const fn standard(type_: ParticleType, p_load: f64, p_savr: f64) -> ParticleDef {
        let load = load_from_imperial(p_load);
        let savr = savr_from_imperial(p_savr);
        let density = density_from_imperial(32.0);
        let heat = heat_from_imperial(8000.0);
        let si_total = mk_ratio(0.0555);
        let si_effective = mk_ratio(0.01);
        ParticleDef {
            type_,
            load,
            savr,
            density,
            heat,
            si_total,
            si_effective,
        }
    }
    const fn is_sentinel(&self) -> bool {
        match self.type_ {
            ParticleType::NoParticle => true,
            _ => false,
        }
    }
    const fn size_class(&self) -> SizeClass {
        let savr = savr_to_imperial(&self.savr);
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
    const fn life(&self) -> Life {
        match &self.type_ {
            ParticleType::Dead => Life::Dead,
            _ => Life::Alive,
        }
    }
    const fn area_weight<'a>(&self, particles: &ParticleDefs) -> f64 {
        const fn fun(p: &ParticleDef, life: &Life) -> f64 {
            match (p.life(), life) {
                (Life::Alive, Life::Alive) | (Life::Dead, Life::Dead) => p.surface_area(),
                _ => 0.0,
            }
        }
        let life = &self.life();
        let total = accum_particles!(particles, fun, life);
        safe_div(self.surface_area(), total)
    }
    const fn size_class_weight(&self, particles: &ParticleDefs) -> f64 {
        const fn fun(
            p: &ParticleDef,
            life: Life,
            sz_class: SizeClass,
            particles: &ParticleDefs,
        ) -> f64 {
            match (p.life(), life) {
                (Life::Alive, Life::Alive) | (Life::Dead, Life::Dead) => {
                    match (p.size_class(), sz_class) {
                        (SizeClass::SC0, SizeClass::SC0)
                        | (SizeClass::SC1, SizeClass::SC1)
                        | (SizeClass::SC2, SizeClass::SC2)
                        | (SizeClass::SC3, SizeClass::SC3)
                        | (SizeClass::SC4, SizeClass::SC4)
                        | (SizeClass::SC5, SizeClass::SC5) => p.area_weight(particles),
                        _ => 0.0,
                    }
                }
                _ => 0.0,
            }
        }
        let life = self.life();
        let size_class = self.size_class();
        accum_particles!(particles, fun, life, size_class, particles)
    }
    const fn surface_area(&self) -> f64 {
        let load = load_to_imperial(&self.load);
        let savr = savr_to_imperial(&self.savr);
        let density = density_to_imperial(&self.density);
        safe_div(load * savr, density)
    }
    const fn sigma_factor(&self) -> f64 {
        let savr = savr_to_imperial(&self.savr);
        SoftF64(safe_div(-138.0, savr)).exp().to_f64()
    }
}

impl Particle {
    const SENTINEL: Self =
        Particle::make(&ParticleDef::SENTINEL, &init_arr(ParticleDef::SENTINEL, []));
    pub const fn make(def: &ParticleDef, particles: &ParticleDefs) -> Particle {
        let ParticleDef {
            type_,
            load,
            savr,
            density,
            heat,
            si_total,
            si_effective,
        } = def;
        let load = load_to_imperial(load);
        let savr = savr_to_imperial(savr);
        let density = density_to_imperial(density);
        let heat = heat_to_imperial(heat);
        let si_total = extract_ratio(si_total);
        let si_effective = extract_ratio(si_effective);
        Particle {
            type_: *type_,
            life: def.life(),
            load,
            savr,
            density,
            heat,
            si_total,
            si_effective,
            area_weight: def.area_weight(particles),
            size_class_weight: def.size_class_weight(particles),
            size_class: def.size_class(),
            surface_area: def.surface_area(),
            sigma_factor: def.sigma_factor(),
        }
    }
    const fn is_sentinel(&self) -> bool {
        match self.type_ {
            ParticleType::NoParticle => true,
            _ => false,
        }
    }
    const fn moisture(&self, terrain: &Terrain) -> f64 {
        extract_ratio(&match self.type_ {
            ParticleType::NoParticle => mk_ratio(9.0E100),
            ParticleType::Herb => terrain.herb,
            ParticleType::Wood => terrain.wood,
            ParticleType::Dead => match self.size_class {
                SizeClass::SC0 | SizeClass::SC1 => terrain.d1hr,
                SizeClass::SC2 | SizeClass::SC3 => terrain.d10hr,
                SizeClass::SC4 | SizeClass::SC5 => terrain.d100hr,
            },
        })
    }
}

impl FuelDef {
    const fn life_particles(&self, life: Life) -> &Particles {
        match life {
            Life::Alive => &self.alive_particles,
            Life::Dead => &self.dead_particles,
        }
    }
    const fn total_area(&self) -> f64 {
        const fn fun(p: &Particle) -> f64 {
            p.surface_area
        }
        accum_particles!(self.life_particles(Life::Alive), fun)
            + accum_particles!(self.life_particles(Life::Dead), fun)
    }
    const fn life_area_weight(&self, life: Life) -> f64 {
        const fn fun(p: &Particle, total_area: f64) -> f64 {
            p.surface_area / total_area
        }
        let total_area = self.total_area();
        accum_particles!(self.life_particles(life), fun, total_area)
    }
    const fn life_fine_load(&self, life: Life) -> f64 {
        const fn fun_alive(p: &Particle) -> f64 {
            p.load * SoftF64(-500.0).div(SoftF64(p.savr)).exp().to_f64()
        }
        const fn fun_dead(p: &Particle) -> f64 {
            p.load * p.sigma_factor
        }
        match life {
            Life::Alive => accum_particles!(self.life_particles(life), fun_alive),
            Life::Dead => accum_particles!(self.life_particles(life), fun_dead),
        }
    }
    const fn life_load(&self, life: Life) -> f64 {
        const fn fun(p: &Particle) -> f64 {
            p.size_class_weight * p.load * (1.0 - p.si_total)
        }
        accum_particles!(self.life_particles(life), fun)
    }
    const fn life_savr(&self, life: Life) -> f64 {
        const fn fun(p: &Particle) -> f64 {
            p.area_weight * p.savr
        }
        accum_particles!(self.life_particles(life), fun)
    }
    const fn life_heat(&self, life: Life) -> f64 {
        const fn fun(p: &Particle) -> f64 {
            p.area_weight * p.heat
        }
        accum_particles!(self.life_particles(life), fun)
    }
    const fn life_seff(&self, life: Life) -> f64 {
        const fn fun(p: &Particle) -> f64 {
            p.area_weight * p.si_effective
        }
        accum_particles!(self.life_particles(life), fun)
    }
    const fn life_eta_s(&self, life: Life) -> f64 {
        let seff: f64 = self.life_seff(life);
        if seff > SMIDGEN {
            let eta = 0.174 / SoftF64(seff).powf(SoftF64(0.19)).to_f64();
            if eta < 1.0 {
                eta
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
    const fn sigma(&self) -> f64 {
        self.life_area_weight(Life::Alive) * self.life_savr(Life::Alive)
            + self.life_area_weight(Life::Dead) * self.life_savr(Life::Dead)
    }
    const fn ratio(&self, sigma: f64, beta: f64) -> f64 {
        beta / (3.348 / SoftF64(sigma).powf(SoftF64(0.8189)).to_f64())
    }
    const fn flux_ratio(&self, sigma: f64, beta: f64) -> f64 {
        ((SoftF64(0.792).add(SoftF64(0.681).mul(SoftF64(sigma).sqrt())))
            .mul(SoftF64(beta).add(SoftF64(0.1))))
        .exp()
        .to_f64()
            / (192.0 + 0.2595 * sigma)
    }
    const fn beta(&self) -> f64 {
        const fn fun(p: &Particle) -> f64 {
            safe_div(p.load, p.density)
        }
        let r = accum_particles!(self.life_particles(Life::Alive), fun)
            + accum_particles!(self.life_particles(Life::Dead), fun);
        safe_div(r, length_to_imperial(&self.depth))
    }
    const fn gamma(&self, sigma: f64, beta: f64) -> f64 {
        let rt = SoftF64(self.ratio(sigma, beta));
        let sigma15 = SoftF64(sigma).powf(SoftF64(1.5));
        let gamma_max = sigma15.div(SoftF64(495.0).add(SoftF64(0.0594).mul(sigma15)));
        let aa = SoftF64(133.0).div(SoftF64(sigma).powf(SoftF64(0.7913)));
        gamma_max
            .mul(rt.powf(aa))
            .mul(aa.mul(SoftF64(1.0).sub(rt)).exp())
            .to_f64()
    }

    const fn life_rx_factor(&self, life: Life, sigma: f64, beta: f64) -> f64 {
        self.life_load(life)
            * self.life_heat(life)
            * self.life_eta_s(life)
            * self.gamma(sigma, beta)
    }

    const fn live_ext_factor(&self) -> f64 {
        2.9 * safe_div(
            self.life_fine_load(Life::Dead),
            self.life_fine_load(Life::Alive),
        )
    }

    const fn bulk_density(&self) -> f64 {
        const fn fun(p: &Particle, depth: f64) -> f64 {
            safe_div(p.load, depth)
        }
        let depth = length_to_imperial(&self.depth);
        accum_particles!(self.life_particles(Life::Alive), fun, depth)
            + accum_particles!(self.life_particles(Life::Dead), fun, depth)
    }

    const fn residence_time(&self, sigma: f64) -> f64 {
        384.0 / sigma
    }

    const fn slope_k(&self, beta: f64) -> f64 {
        5.275 * SoftF64(beta).powf(SoftF64(-0.3)).to_f64()
    }

    const fn wind_bke(&self, sigma: f64, beta: f64) -> (f64, f64, f64) {
        let wind_b = 0.02526 * SoftF64(sigma).powf(SoftF64(0.54)).to_f64();
        let r = self.ratio(sigma, beta);
        let c = 7.47
            * (SoftF64(-0.133).mul(SoftF64(sigma).powf(SoftF64(0.55))))
                .exp()
                .to_f64();
        let e = 0.715 * SoftF64((-0.000359) * sigma).exp().to_f64();
        let wind_k = c * SoftF64(r).powf(SoftF64(-e)).to_f64();
        let wind_e = SoftF64(r).powf(SoftF64(e)).to_f64() / c;
        (wind_b, wind_k, wind_e)
    }
}

impl Fuel {
    pub(crate) const SENTINEL: Self = Self::standard(b"", b"", 0.0, 0.0, []);

    pub const fn standard<const N: usize, const M: usize, const F: usize>(
        name: &[u8; N],
        desc: &[u8; M],
        depth: f64,
        mext: f64,
        particles: [ParticleDef; F],
    ) -> Self {
        let particles = init_arr(ParticleDef::SENTINEL, particles);
        let mut alive_particles = [Particle::SENTINEL; MAX_PARTICLES];
        let mut dead_particles = [Particle::SENTINEL; MAX_PARTICLES];
        let mut i = 0;
        let mut i_a = 0;
        let mut i_d = 0;
        while i < MAX_PARTICLES {
            let p = &particles[i];
            if !p.is_sentinel() {
                let p2 = Particle::make(p, &particles);
                match p.life() {
                    Life::Alive => {
                        alive_particles[i_a] = p2;
                        i_a += 1;
                    }
                    Life::Dead => {
                        dead_particles[i_d] = p2;
                        i_d += 1;
                    }
                };
                i += 1
            } else {
                break;
            };
        }
        Self::make(FuelDef {
            name: init_arr(0, *name),
            desc: init_arr(0, *desc),
            depth: length_from_imperial(depth),
            mext: mk_ratio(mext),
            adjust: mk_ratio(1.0),
            alive_particles,
            dead_particles,
        })
    }

    pub const fn make(fuel: FuelDef) -> Fuel {
        let depth = length_to_imperial(&fuel.depth);
        let mext = extract_ratio(&fuel.mext);
        let adjust = extract_ratio(&fuel.adjust);
        let sigma = fuel.sigma();
        let beta = fuel.beta();
        let (wind_b, wind_k, wind_e) = fuel.wind_bke(sigma, beta);
        let life_rx_factor_alive = fuel.life_rx_factor(Life::Alive, sigma, beta);
        let life_rx_factor_dead = fuel.life_rx_factor(Life::Dead, sigma, beta);
        let total_area = fuel.total_area();
        Self {
            name: fuel.name,
            desc: fuel.desc,
            adjust,
            depth,
            live_area_weight: fuel.life_area_weight(Life::Alive),
            live_rx_factor: fuel.life_rx_factor(Life::Alive, sigma, beta),
            dead_area_weight: fuel.life_area_weight(Life::Dead),
            dead_rx_factor: fuel.life_rx_factor(Life::Dead, sigma, beta),
            fine_dead_factor: fuel.life_fine_load(Life::Dead),
            live_ext_factor: fuel.live_ext_factor(),
            fuel_bed_bulk_dens: fuel.bulk_density(),
            residence_time: fuel.residence_time(sigma),
            flux_ratio: fuel.flux_ratio(sigma, beta),
            slope_k: fuel.slope_k(beta),
            mext,
            total_area,
            wind_b,
            wind_e,
            wind_k,
            alive_particles: fuel.alive_particles,
            dead_particles: fuel.dead_particles,
            sigma,
            beta,
            life_rx_factor_alive,
            life_rx_factor_dead,
        }
    }
    const fn life_particles(&self, life: Life) -> &Particles {
        match life {
            Life::Alive => &self.alive_particles,
            Life::Dead => &self.dead_particles,
        }
    }
    const fn has_particles(&self) -> bool {
        self.has_live_particles() || self.has_dead_particles()
    }
    const fn has_live_particles(&self) -> bool {
        !self.alive_particles[0].is_sentinel()
    }
    const fn has_dead_particles(&self) -> bool {
        !self.dead_particles[0].is_sentinel()
    }
    const fn life_area_weight(&self, life: Life) -> f64 {
        match life {
            Life::Alive => self.live_area_weight,
            Life::Dead => self.dead_area_weight,
        }
    }

    pub fn burn(&self, terrain: &Terrain) -> Option<Fire> {
        if !self.has_particles() {
            None
        } else {
            let rx_int = self.rx_int(terrain);
            let speed0 = self.speed0(terrain, rx_int);
            let (phi_eff_wind, eff_wind, speed_max, azimuth_max) =
                self.calculate_wind_dependent_vars(terrain, speed0, rx_int);
            Some(Fire {
                rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(rx_int),
                speed0: Velocity::new::<foot_per_minute>(speed0),
                hpua: RadiantExposure::new::<btu_sq_foot>(self.hpua(rx_int)),
                phi_eff_wind: Ratio::new::<ratio>(phi_eff_wind),
                speed_max: Velocity::new::<foot_per_minute>(speed_max),
                azimuth_max: Angle::new::<radian>(azimuth_max),
                eccentricity: Ratio::new::<ratio>(Self::eccentricity(eff_wind)),
                residence_time: Time::new::<minute>(self.residence_time),
            })
        }
    }
    fn particles(&self) -> impl Iterator<Item = &Particle> {
        iter_particles(&self.dead_particles).chain(iter_particles(&self.alive_particles))
    }
    fn rx_int(&self, terrain: &Terrain) -> f64 {
        self.life_rx_factor_alive * self.life_eta_m(Life::Alive, terrain)
            + self.life_rx_factor_dead * self.life_eta_m(Life::Dead, terrain)
    }
    fn calculate_wind_dependent_vars(
        &self,
        terrain: &Terrain,
        speed0: f64,
        rx_int: f64,
    ) -> (f64, f64, f64, f64) {
        let phi_ew = self.phi_ew(terrain);
        let speed_max1 = speed0 * (1.0 + phi_ew);
        let upslope = terrain.upslope();
        let wind_speed = terrain.wind_speed.get::<foot_per_minute>();
        let wind_az = terrain.wind_azimuth.get::<radian>();
        let ew_from_phi_ew = |p: f64| f64::powf(p * self.wind_e, 1.0 / self.wind_b);
        let max_wind = 0.9 * rx_int;
        let check_wind_limit = |pew: f64, ew: f64, s: f64, a: f64| {
            if ew > max_wind {
                let phi_ew_max_wind = if max_wind < SMIDGEN {
                    0.0
                } else {
                    self.wind_k * f64::powf(max_wind, self.wind_b)
                };
                let speed_max_wind = speed0 * (1.0 + phi_ew_max_wind);
                (phi_ew_max_wind, max_wind, speed_max_wind, a)
            } else {
                (pew, ew, s, a)
            }
        };
        use WindSlopeSituation::*;
        match self.wind_slope_situation(terrain, speed0, phi_ew) {
            NoSpread => (phi_ew, 0.0, 0.0, 0.0),
            NoSlopeNoWind => (phi_ew, 0.0, speed0, 0.0),
            WindNoSlope => check_wind_limit(phi_ew, wind_speed, speed_max1, wind_az),
            SlopeNoWind | UpSlope => {
                check_wind_limit(phi_ew, ew_from_phi_ew(phi_ew), speed_max1, upslope)
            }
            CrossSlope => {
                let wind_rate = speed0 * self.phi_wind(terrain);
                let slope_rate = speed0 * self.phi_slope(terrain);
                let split = if upslope <= wind_az {
                    wind_az - upslope
                } else {
                    2.0 * PI - upslope + wind_az
                };
                let x = slope_rate + wind_rate * f64::cos(split);
                let y = wind_rate * f64::sin(split);
                let rv = f64::sqrt(x * x + y * y);
                let speed_max2 = speed0 + rv;
                let phi_ew2 = speed_max2 / speed0 - 1.0;
                let eff_wind = {
                    if phi_ew2 > SMIDGEN {
                        ew_from_phi_ew(phi_ew2)
                    } else {
                        0.0
                    }
                };
                let al = f64::asin(y.abs() / rv);
                let split2 = {
                    if x >= 0.0 && y >= 0.0 {
                        al
                    } else if x >= 0.0 {
                        2.0 * PI - al
                    } else if y >= 0.0 {
                        PI - al
                    } else {
                        PI + al
                    }
                };
                let azimuth_max2 = {
                    let ret = upslope + split2;
                    if ret > 2.0 * PI {
                        ret - 2.0 * PI
                    } else {
                        ret
                    }
                };
                check_wind_limit(phi_ew2, eff_wind, speed_max2, azimuth_max2)
            }
        }
    }
    fn wind_slope_situation(
        &self,
        terrain: &Terrain,
        speed0: f64,
        phi_ew: f64,
    ) -> WindSlopeSituation {
        let wind_az = terrain.wind_azimuth.get::<radian>();

        use WindSlopeSituation::*;
        if speed0 < SMIDGEN {
            NoSpread
        } else if phi_ew < SMIDGEN {
            NoSlopeNoWind
        } else if terrain.slope.get::<ratio>() < SMIDGEN {
            WindNoSlope
        } else if terrain.wind_speed.get::<meter_per_second>() < SMIDGEN {
            SlopeNoWind
        } else if (terrain.upslope() - wind_az).abs() < SMIDGEN {
            UpSlope
        } else {
            CrossSlope
        }
    }
    fn speed0(&self, terrain: &Terrain, rx_int: f64) -> f64 {
        safe_div(rx_int * self.flux_ratio, self.rbqig(terrain))
    }
    fn rbqig(&self, terrain: &Terrain) -> f64 {
        let x: f64 = self
            .particles()
            .map(|p| {
                (250.0 + 1116.0 * p.moisture(terrain))
                    * p.area_weight
                    * self.life_area_weight(p.life)
                    * p.sigma_factor
            })
            .sum();
        self.fuel_bed_bulk_dens * x
    }
    fn hpua(&self, rx_int: f64) -> f64 {
        rx_int * self.residence_time
    }
    fn eccentricity(eff_wind: f64) -> f64 {
        let lw_ratio = 1.0 + 0.002840909 * eff_wind;
        if eff_wind > SMIDGEN && lw_ratio > 1.0 + SMIDGEN {
            f64::sqrt(lw_ratio * lw_ratio - 1.0) / lw_ratio
        } else {
            0.0
        }
    }
    fn life_moisture(&self, life: Life, terrain: &Terrain) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.area_weight * p.moisture(terrain))
            .sum()
    }

    fn life_mext(&self, life: Life, terrain: &Terrain) -> f64 {
        match life {
            Life::Alive => {
                if self.has_live_particles() {
                    let fdmois = safe_div(self.wfmd(terrain), self.fine_dead_factor);
                    let live_mext = self.live_ext_factor * (1.0 - fdmois / self.mext) - 0.226;
                    live_mext.max(self.mext)
                } else {
                    0.0
                }
            }
            Life::Dead => self.mext,
        }
    }

    fn life_eta_m(&self, life: Life, terrain: &Terrain) -> f64 {
        let life_mext = self.life_mext(life, terrain);
        let life_moist = self.life_moisture(life, terrain);
        if life_moist >= life_mext {
            0.0
        } else if life_mext > SMIDGEN {
            let rt = life_moist / life_mext;
            1.0 - 2.59 * rt + 5.11 * rt * rt - 3.52 * rt * rt * rt
        } else {
            0.0
        }
    }

    fn wfmd(&self, terrain: &Terrain) -> f64 {
        iter_particles(self.life_particles(Life::Dead))
            .map(|p| p.moisture(terrain) * p.sigma_factor * p.load)
            .sum()
    }

    fn phi_slope(&self, terrain: &Terrain) -> f64 {
        let s = terrain.slope.get::<ratio>();
        self.slope_k * s * s
    }

    fn phi_wind(&self, terrain: &Terrain) -> f64 {
        let ws = terrain.wind_speed.get::<foot_per_minute>();
        if ws > SMIDGEN {
            self.wind_k * f64::powf(ws, self.wind_b)
        } else {
            0.0
        }
    }

    fn phi_ew(&self, terrain: &Terrain) -> f64 {
        self.phi_slope(terrain) + self.phi_wind(terrain)
    }
}

const fn safe_div(a: f64, b: f64) -> f64 {
    if b > SMIDGEN {
        a / b
    } else {
        0.0
    }
}

#[derive(Debug)]
enum WindSlopeSituation {
    NoSpread,
    NoSlopeNoWind,
    WindNoSlope,
    SlopeNoWind,
    UpSlope,
    CrossSlope,
}

fn iter_particles<const N: usize>(particles: &[Particle; N]) -> impl Iterator<Item = &Particle> {
    particles.iter().take_while(|p| !p.is_sentinel())
}
const fn init_arr<T: Copy, const N: usize, const M: usize>(def: T, src: [T; M]) -> [T; N] {
    let mut dst = [def; N];
    let mut i = 0;
    while i < M {
        dst[i] = src[i];
        i += 1;
    }
    dst
}

#[cfg(target_os = "cuda")]
mod f64 {
    pub(crate) fn powf(x: f64, y: f64) -> f64 {
        use cuda_std::intrinsics::*;
        unsafe { pow(x, y) }
    }
    pub(crate) fn cos(x: f64) -> f64 {
        use cuda_std::intrinsics::*;
        unsafe { cos(x) }
    }
    pub(crate) fn sin(x: f64) -> f64 {
        use cuda_std::intrinsics::*;
        unsafe { sin(x) }
    }
    pub(crate) fn asin(x: f64) -> f64 {
        use cuda_std::intrinsics::*;
        unsafe { asin(x) }
    }
    pub(crate) fn sqrt(x: f64) -> f64 {
        use cuda_std::intrinsics::*;
        unsafe { sqrt(x) }
    }
}

