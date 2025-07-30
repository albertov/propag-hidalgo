#[cfg(not(target_os = "cuda"))]
extern crate std;

#[cfg(not(target_os = "cuda"))]
use std::vec::Vec;

#[cfg(target_os = "cuda")]
extern crate cuda_std;

#[cfg(not(target_os = "cuda"))]
extern crate cust_core;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

use crate::units::heat_flux_density::btu_sq_foot_min;
use crate::units::linear_power_density::btu_foot_sec;
use crate::units::radiant_exposure::btu_sq_foot;
use crate::units::*;
use uom::si::angle::{degree, radian};
use uom::si::length::foot;
use uom::si::ratio::ratio;
use uom::si::time::minute;
use uom::si::velocity::foot_per_minute;
use uom::si::velocity::meter_per_second;

use crate::float;
use crate::float::*;

const MAX_PARTICLES: usize = 5;
const MAX_FUELS: usize = 20;

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub enum Life {
    Dead,
    Alive,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum ParticleType {
    Dead,       // Dead fuel particle
    Herb,       // Herbaceous live particle
    Wood,       // Woody live particle
    NoParticle, // Sentinel for no particle
}

#[derive(Clone, Copy)]
#[repr(C)]
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
#[repr(C)]
pub struct Particle {
    pub type_: ParticleType,
    pub load: float::T,
    pub savr: float::T,
    pub density: float::T,
    pub heat: float::T,
    pub si_total: float::T,
    pub si_effective: float::T,
    pub area_weight: float::T,
    pub surface_area: float::T,
    pub sigma_factor: float::T,
    pub size_class_weight: float::T,
    pub size_class: SizeClass,
    pub life: Life,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_os = "cuda"), derive(StructOfArray), soa_derive(Debug))]
#[repr(C)]
pub struct Terrain {
    pub fuel_code: u8,
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

#[cfg_attr(not(target_os = "cuda"), derive(StructOfArray), soa_derive(Debug))]
#[repr(C)]
#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
pub struct TerrainCuda {
    pub fuel_code: u8,
    pub d1hr: float::T,
    pub d10hr: float::T,
    pub d100hr: float::T,
    pub herb: float::T,
    pub wood: float::T,
    pub wind_speed: float::T,
    pub wind_azimuth: float::T,
    pub slope: float::T,
    pub aspect: float::T,
}

impl TerrainCuda {
    pub const NULL: Self = Self {
        fuel_code: 0,
        d1hr: 0.0,
        d10hr: 0.0,
        d100hr: 0.0,
        herb: 0.0,
        wood: 0.0,
        wind_speed: 0.0,
        wind_azimuth: 0.0,
        slope: 0.0,
        aspect: 0.0,
    };
}

#[macro_export]
macro_rules! to_quantity {
    ($quant:ident, $val:expr) => {{
        use uom::lib::marker::PhantomData;
        $quant {
            value: $val,
            units: PhantomData,
            dimension: PhantomData,
        }
    }};
}
#[macro_export]
macro_rules! from_quantity {
    ($quant:ident, $val:expr) => {{
        let $quant { value, .. } = $val;
        *value
    }};
}
impl From<TerrainCuda> for Terrain {
    fn from(f: TerrainCuda) -> Self {
        Self {
            fuel_code: f.fuel_code,
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
            fuel_code: f.fuel_code,
            d1hr: from_quantity!(Ratio, &f.d1hr),
            d10hr: from_quantity!(Ratio, &f.d10hr),
            d100hr: from_quantity!(Ratio, &f.d100hr),
            herb: from_quantity!(Ratio, &f.herb),
            wood: from_quantity!(Ratio, &f.wood),
            wind_speed: from_quantity!(Velocity, &f.wind_speed),
            wind_azimuth: from_quantity!(Angle, &f.wind_azimuth),
            slope: from_quantity!(Ratio, &f.slope),
            aspect: from_quantity!(Angle, &f.aspect),
        }
    }
}
#[cfg(not(target_os = "cuda"))]
impl From<TerrainCudaRef<'_>> for Terrain {
    fn from(f: TerrainCudaRef<'_>) -> Self {
        From::<TerrainCuda>::from(f.into())
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
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
#[cfg_attr(not(target_os = "cuda"), derive(StructOfArray), soa_derive(Debug))]
#[repr(C)]
#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
pub struct FireCuda {
    pub rx_int: float::T,
    pub speed0: float::T,
    pub hpua: float::T,
    pub phi_eff_wind: float::T,
    pub speed_max: float::T,
    pub azimuth_max: float::T,
    pub eccentricity: float::T,
    pub residence_time: float::T,
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
            rx_int: from_quantity!(HeatFluxDensity, &f.rx_int),
            speed0: from_quantity!(Velocity, &f.speed0),
            hpua: from_quantity!(RadiantExposure, &f.hpua),
            phi_eff_wind: from_quantity!(Ratio, &f.phi_eff_wind),
            speed_max: from_quantity!(Velocity, &f.speed_max),
            azimuth_max: from_quantity!(Angle, &f.azimuth_max),
            eccentricity: from_quantity!(Ratio, &f.eccentricity),
            residence_time: from_quantity!(Time, &f.residence_time),
        }
    }
}
#[cfg(not(target_os = "cuda"))]
impl From<FireCudaPtr> for Fire {
    fn from(f: FireCudaPtr) -> Self {
        unsafe { f.read().into() }
    }
}
#[cfg(not(target_os = "cuda"))]
impl From<FireCudaRef<'_>> for Fire {
    fn from(f: FireCudaRef<'_>) -> Self {
        From::<FireCuda>::from(f.into())
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct FireSimple {
    pub speed_max: Velocity,
    pub azimuth_max: Angle,
    pub eccentricity: Ratio,
}

#[derive(Debug)]
#[repr(C)]
pub struct Spread<'a, T> {
    fire: &'a T,
    factor: float::T,
}

pub trait CanSpread<'a> {
    fn azimuth_max(&self) -> Angle;
    fn eccentricity(&self) -> Ratio;

    fn spread(&'a self, azimuth: Angle) -> Spread<'a, Self>
    where
        Self: Sized,
    {
        let azimuth = azimuth.get::<radian>();
        let azimuth_max = self.azimuth_max().get::<radian>();
        let angle = (azimuth - azimuth_max).abs();
        let ecc = self.eccentricity().get::<ratio>();
        let factor = (1.0 - ecc) / (1.0 - ecc * angle.cos());
        Spread { fire: self, factor }
    }
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

#[repr(C)]
pub struct FuelDef {
    pub name: [u8; 16],
    pub desc: [u8; 64],
    pub depth: Length,
    pub mext: Ratio,
    pub adjust: Ratio,
    pub particles: Particles,
}

pub type ParticleDefs = [ParticleDef; MAX_PARTICLES];

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Fuel {
    pub name: [u8; 16],
    pub desc: [u8; 64],
    pub depth: float::T,
    pub mext: float::T,
    pub inv_mext: float::T,
    pub adjust: float::T,
    pub n_dead_particles: usize,
    pub particles: Particles,
    pub live_area_weight: float::T,
    pub live_rx_factor: float::T,
    pub dead_area_weight: float::T,
    pub dead_rx_factor: float::T,
    pub fine_dead_factor: float::T,
    pub live_ext_factor: float::T,
    pub fuel_bed_bulk_dens: float::T,
    pub residence_time: float::T,
    pub flux_ratio: float::T,
    pub slope_k: float::T,
    pub wind_b: float::T,
    pub wind_b_inv: float::T,
    pub wind_e: float::T,
    pub wind_k: float::T,
    pub sigma: float::T,
    pub beta: float::T,
    pub life_rx_factor_alive: float::T,
    pub life_rx_factor_dead: float::T,
    pub total_area: float::T,
}

pub type Particles = [Particle; MAX_PARTICLES];

#[repr(C)]
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
    #[inline]
    pub fn ignite(&self, terrain: &Terrain) -> Option<Fire> {
        self.get(terrain.fuel_code as _)
            .and_then(|f| f.burn(terrain))
    }
    #[inline]
    pub fn burn_simple(&self, terrain: &Terrain) -> Option<FireSimple> {
        self.get(terrain.fuel_code as _)
            .and_then(|f| f.burn_simple(terrain))
    }
}

impl Terrain {
    fn upslope(&self) -> float::T {
        let aspect = self.aspect.get::<radian>();
        if aspect >= PI {
            aspect - PI
        } else {
            aspect + PI
        }
    }
}

impl Fire {
    pub const NULL: Self = {
        Self {
            rx_int: to_quantity!(HeatFluxDensity, 0.0),
            speed0: to_quantity!(Velocity, 0.0),
            hpua: to_quantity!(RadiantExposure, 0.0),
            phi_eff_wind: to_quantity!(Ratio, 0.0),
            speed_max: to_quantity!(Velocity, 0.0),
            azimuth_max: to_quantity!(Angle, 0.0),
            eccentricity: to_quantity!(Ratio, 0.0),
            residence_time: to_quantity!(Time, 0.0),
        }
    };
    fn flame_length(byrams_intensity: float::T) -> float::T {
        if byrams_intensity > SMIDGEN {
            0.45 * byrams_intensity.powf(0.46)
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

    pub fn almost_eq(&self, other: &Self) -> bool {
        fuzzy_cmp_smidgen(
            "rx_int",
            self.rx_int.get::<btu_sq_foot_min>(),
            other.rx_int.get::<btu_sq_foot_min>(),
            1e-1,
        ) && fuzzy_cmp(
            "speed0",
            self.speed0.get::<foot_per_minute>(),
            other.speed0.get::<foot_per_minute>(),
        ) && fuzzy_cmp(
            "hpua",
            self.hpua.get::<btu_sq_foot>(),
            other.hpua.get::<btu_sq_foot>(),
        ) && fuzzy_cmp(
            "speed_max",
            self.speed_max.get::<foot_per_minute>(),
            other.speed_max.get::<foot_per_minute>(),
        ) && fuzzy_cmp(
            "eccentricity",
            self.eccentricity.get::<ratio>(),
            other.eccentricity.get::<ratio>(),
        ) && fuzzy_cmp(
            "byrams_max",
            self.byrams_max().get::<btu_foot_sec>(),
            other.byrams_max().get::<btu_foot_sec>(),
        ) && fuzzy_cmp(
            "flame_max",
            self.flame_max().get::<foot>(),
            other.flame_max().get::<foot>(),
        ) && (
            // Sometimes when float::T is f32 the NoSpread branch
            // situation does not happen while in fireLib (C) which
            // uses f64 does. In that situation fireLib sets azimuth to
            // 0.0. If speed_max is small we don't care about this
            // discrepancy
            fuzzy_cmp_azimuths("azimuth_max", self.azimuth_max, other.azimuth_max) || {
                #[cfg(not(target_os = "cuda"))]
                {
                    use std::println;
                    println!(
                        "azimuth_max={:?}, speed_max={:?}",
                        self.azimuth_max, self.speed_max
                    );
                };
                self.speed_max.get::<foot_per_minute>() < 1.0
            }
        )
    }
}
impl FireSimple {
    pub const NULL: Self = {
        Self {
            speed_max: to_quantity!(Velocity, 0.0),
            azimuth_max: to_quantity!(Angle, 0.0),
            eccentricity: to_quantity!(Ratio, 0.0),
        }
    };
    pub fn almost_eq(&self, other: &Self) -> bool {
        fuzzy_cmp(
            "speed_max",
            self.speed_max.get::<foot_per_minute>(),
            other.speed_max.get::<foot_per_minute>(),
        ) && fuzzy_cmp(
            "eccentricity",
            self.eccentricity.get::<ratio>(),
            other.eccentricity.get::<ratio>(),
        ) && fuzzy_cmp(
            "azimuth_max",
            self.azimuth_max.get::<radian>(),
            other.azimuth_max.get::<radian>(),
        )
    }
}

impl CanSpread<'_> for Fire {
    fn azimuth_max(&self) -> Angle {
        self.azimuth_max
    }
    fn eccentricity(&self) -> Ratio {
        self.eccentricity
    }
}

impl CanSpread<'_> for FireSimple {
    fn azimuth_max(&self) -> Angle {
        self.azimuth_max
    }
    fn eccentricity(&self) -> Ratio {
        self.eccentricity
    }
}

impl Spread<'_, FireSimple> {
    pub fn speed(&self) -> Velocity {
        self.fire.speed_max * to_quantity!(Ratio, self.factor)
    }
}

impl Spread<'_, Fire> {
    pub fn speed(&self) -> Velocity {
        self.fire.speed_max * to_quantity!(Ratio, self.factor)
    }
    pub fn byrams(&self) -> LinearPowerDensity {
        self.fire.byrams_max() * to_quantity!(Ratio, self.factor)
    }
    pub fn flame(&self) -> Length {
        Length::new::<foot>(Fire::flame_length(self.byrams().get::<btu_foot_sec>()))
    }
}

impl ParticleDef {
    const SENTINEL: Self = ParticleDef::standard(ParticleType::NoParticle, 0.0, 0.0);

    pub const fn standard(type_: ParticleType, p_load: float::T, p_savr: float::T) -> ParticleDef {
        let load = load_from_imperial(p_load);
        let savr = savr_from_imperial(p_savr);
        let density = density_from_imperial(32.0);
        let heat = heat_from_imperial(8000.0);
        let si_total = to_quantity!(Ratio, 0.0555);
        let si_effective = to_quantity!(Ratio, 0.01);
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
        matches!(self.type_, ParticleType::NoParticle)
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
    const fn same_life(&self, life: Life) -> bool {
        matches!(
            (self.life(), life),
            (Life::Alive, Life::Alive) | (Life::Dead, Life::Dead)
        )
    }
    const fn area_weight<const N: usize>(&self, particles: &[ParticleDef; N]) -> float::T {
        const fn fun(p: &ParticleDef, life: Life) -> float::T {
            if p.same_life(life) {
                p.surface_area()
            } else {
                0.0
            }
        }
        let life = self.life();
        let total = accum_particles!(particles, fun, life);
        safe_div(self.surface_area(), total)
    }
    const fn size_class_weight<const N: usize>(&self, particles: &[ParticleDef; N]) -> float::T {
        const fn fun<const N: usize>(
            p: &ParticleDef,
            life: Life,
            sz_class: SizeClass,
            particles: &[ParticleDef; N],
        ) -> float::T {
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
    const fn surface_area(&self) -> float::T {
        let load = load_to_imperial(&self.load);
        let savr = savr_to_imperial(&self.savr);
        let density = density_to_imperial(&self.density);
        safe_div(load * savr, density)
    }
    const fn sigma_factor(&self) -> float::T {
        let savr = savr_to_imperial(&self.savr);
        SoftFloat(safe_div(-138.0, savr)).exp().to_float()
    }
}

impl Particle {
    const SENTINEL: Self =
        Particle::make::<0>(&ParticleDef::SENTINEL, &init_arr(ParticleDef::SENTINEL, []));
    pub const fn make<const N: usize>(def: &ParticleDef, particles: &[ParticleDef; N]) -> Particle {
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
        let si_total = from_quantity!(Ratio, si_total);
        let si_effective = from_quantity!(Ratio, si_effective);
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
        matches!(self.type_, ParticleType::NoParticle)
    }
    const fn same_life(&self, life: Life) -> bool {
        matches!(
            (self.life, life),
            (Life::Alive, Life::Alive) | (Life::Dead, Life::Dead)
        )
    }
    const fn moisture(&self, terrain: &Terrain) -> float::T {
        from_quantity!(
            Ratio,
            &match self.type_ {
                ParticleType::NoParticle => to_quantity!(Ratio, 9.0E32),
                ParticleType::Herb => terrain.herb,
                ParticleType::Wood => terrain.wood,
                ParticleType::Dead => match self.size_class {
                    SizeClass::SC0 | SizeClass::SC1 => terrain.d1hr,
                    SizeClass::SC2 | SizeClass::SC3 => terrain.d10hr,
                    SizeClass::SC4 | SizeClass::SC5 => terrain.d100hr,
                },
            }
        )
    }
}

impl FuelDef {
    const fn total_area(&self) -> float::T {
        const fn fun(p: &Particle) -> float::T {
            p.surface_area
        }
        accum_particles!(self.particles, fun)
    }
    const fn life_area_weight(&self, life: Life) -> float::T {
        const fn fun(p: &Particle, life: Life, total_area: float::T) -> float::T {
            if p.same_life(life) {
                p.surface_area / total_area
            } else {
                0.0
            }
        }
        let total_area = self.total_area();
        accum_particles!(self.particles, fun, life, total_area)
    }
    const fn life_fine_load(&self, life: Life) -> float::T {
        const fn fun(p: &Particle, life: Life) -> float::T {
            if !p.same_life(life) {
                return 0.0;
            };
            match life {
                Life::Alive => p.load * SoftFloat(-500.0).div(SoftFloat(p.savr)).exp().to_float(),
                Life::Dead => p.load * p.sigma_factor,
            }
        }
        accum_particles!(self.particles, fun, life)
    }
    const fn life_load(&self, life: Life) -> float::T {
        const fn fun(p: &Particle, life: Life) -> float::T {
            if !p.same_life(life) {
                return 0.0;
            };
            p.size_class_weight * p.load * (1.0 - p.si_total)
        }
        accum_particles!(self.particles, fun, life)
    }
    const fn life_savr(&self, life: Life) -> float::T {
        const fn fun(p: &Particle, life: Life) -> float::T {
            if !p.same_life(life) {
                return 0.0;
            };
            p.area_weight * p.savr
        }
        accum_particles!(self.particles, fun, life)
    }
    const fn life_heat(&self, life: Life) -> float::T {
        const fn fun(p: &Particle, life: Life) -> float::T {
            if !p.same_life(life) {
                return 0.0;
            };
            p.area_weight * p.heat
        }
        accum_particles!(self.particles, fun, life)
    }
    const fn life_seff(&self, life: Life) -> float::T {
        const fn fun(p: &Particle, life: Life) -> float::T {
            if !p.same_life(life) {
                return 0.0;
            };
            p.area_weight * p.si_effective
        }
        accum_particles!(self.particles, fun, life)
    }
    const fn life_eta_s(&self, life: Life) -> float::T {
        let seff: float::T = self.life_seff(life);
        if seff > SMIDGEN {
            let eta = 0.174 / SoftFloat(seff).powf(SoftFloat(0.19)).to_float();
            if eta < 1.0 {
                eta
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
    const fn sigma(&self) -> float::T {
        self.life_area_weight(Life::Alive) * self.life_savr(Life::Alive)
            + self.life_area_weight(Life::Dead) * self.life_savr(Life::Dead)
    }
    const fn ratio(&self, sigma: float::T, beta: float::T) -> float::T {
        beta / (3.348 / SoftFloat(sigma).powf(SoftFloat(0.8189)).to_float())
    }
    const fn flux_ratio(&self, sigma: float::T, beta: float::T) -> float::T {
        ((SoftFloat(0.792).add(SoftFloat(0.681).mul(SoftFloat(sigma).sqrt())))
            .mul(SoftFloat(beta).add(SoftFloat(0.1))))
        .exp()
        .to_float()
            / (192.0 + 0.2595 * sigma)
    }
    const fn beta(&self) -> float::T {
        const fn fun(p: &Particle) -> float::T {
            p.load / p.density
        }
        accum_particles!(self.particles, fun) / length_to_imperial(&self.depth)
    }
    const fn gamma(&self, sigma: float::T, beta: float::T) -> float::T {
        let rt = SoftFloat(self.ratio(sigma, beta));
        let sigma15 = SoftFloat(sigma).powf(SoftFloat(1.5));
        let gamma_max = sigma15.div(SoftFloat(495.0).add(SoftFloat(0.0594).mul(sigma15)));
        let aa = SoftFloat(133.0).div(SoftFloat(sigma).powf(SoftFloat(0.7913)));
        gamma_max
            .mul(rt.powf(aa))
            .mul(aa.mul(SoftFloat(1.0).sub(rt)).exp())
            .to_float()
    }

    const fn life_rx_factor(&self, life: Life, sigma: float::T, beta: float::T) -> float::T {
        self.life_load(life)
            * self.life_heat(life)
            * self.life_eta_s(life)
            * self.gamma(sigma, beta)
    }

    const fn live_ext_factor(&self) -> float::T {
        2.9 * safe_div(
            self.life_fine_load(Life::Dead),
            self.life_fine_load(Life::Alive),
        )
    }

    const fn bulk_density(&self) -> float::T {
        const fn fun(p: &Particle, depth: float::T) -> float::T {
            p.load / depth
        }
        let depth = length_to_imperial(&self.depth);
        accum_particles!(self.particles, fun, depth)
    }

    const fn residence_time(&self, sigma: float::T) -> float::T {
        384.0 / sigma
    }

    const fn slope_k(&self, beta: float::T) -> float::T {
        5.275 * SoftFloat(beta).powf(SoftFloat(-0.3)).to_float()
    }

    const fn wind_bke(&self, sigma: float::T, beta: float::T) -> (float::T, float::T, float::T) {
        let wind_b = 0.02526 * SoftFloat(sigma).powf(SoftFloat(0.54)).to_float();
        let r = self.ratio(sigma, beta);
        let c = 7.47
            * (SoftFloat(-0.133).mul(SoftFloat(sigma).powf(SoftFloat(0.55))))
                .exp()
                .to_float();
        let e = 0.715 * SoftFloat((-0.000359) * sigma).exp().to_float();
        let wind_k = c * SoftFloat(r).powf(SoftFloat(-e)).to_float();
        let wind_e = SoftFloat(r).powf(SoftFloat(e)).to_float() / c;
        (wind_b, wind_k, wind_e)
    }
}

impl Fuel {
    pub(crate) const SENTINEL: Self = Self::standard(b"", b"", 0.0, 0.0, []);

    pub const fn standard<const N: usize, const M: usize, const F: usize>(
        name: &[u8; N],
        desc: &[u8; M],
        depth: float::T,
        mext: float::T,
        particles: [ParticleDef; F],
    ) -> Self {
        if MAX_PARTICLES < particles.len() {
            panic!("bad")
        }
        let mut sorted_particles = [Particle::SENTINEL; MAX_PARTICLES];
        let mut i = 0;
        let mut n_dead_particles = 0;
        while i < particles.len() {
            let p = &particles[i];
            if !p.is_sentinel() {
                #[allow(clippy::single_match)]
                match p.life() {
                    Life::Dead => {
                        sorted_particles[n_dead_particles] = Particle::make(p, &particles);
                        n_dead_particles += 1;
                    }
                    _ => (),
                };
                i += 1
            } else {
                break;
            };
        }
        i = 0;
        let mut j = n_dead_particles;
        while i < particles.len() {
            let p = &particles[i];
            if !p.is_sentinel() {
                #[allow(clippy::single_match)]
                match p.life() {
                    Life::Alive => {
                        sorted_particles[j] = Particle::make(p, &particles);
                        j += 1;
                    }
                    _ => (),
                };
                i += 1;
            } else {
                break;
            };
        }
        Self::make(FuelDef {
            name: init_arr(0, *name),
            desc: init_arr(0, *desc),
            depth: length_from_imperial(depth),
            mext: to_quantity!(Ratio, mext),
            adjust: to_quantity!(Ratio, 1.0),
            particles: sorted_particles,
        })
    }

    pub const fn make(fuel: FuelDef) -> Fuel {
        let depth = length_to_imperial(&fuel.depth);
        let mext = from_quantity!(Ratio, &fuel.mext);
        let adjust = from_quantity!(Ratio, &fuel.adjust);
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
            fine_dead_factor: 1.0 / fuel.life_fine_load(Life::Dead),
            live_ext_factor: fuel.live_ext_factor(),
            fuel_bed_bulk_dens: fuel.bulk_density(),
            residence_time: fuel.residence_time(sigma),
            flux_ratio: fuel.flux_ratio(sigma, beta),
            slope_k: fuel.slope_k(beta),
            mext,
            inv_mext: 1.0 / mext,
            total_area,
            wind_b,
            wind_b_inv: 1.0 / wind_b,
            wind_e,
            wind_k,
            particles: fuel.particles,
            n_dead_particles: {
                let mut i = 0;
                while i < MAX_PARTICLES {
                    let p = &fuel.particles[i];
                    if p.is_sentinel() || p.same_life(Life::Alive) {
                        break;
                    }
                    i += 1
                }
                i
            },
            sigma,
            beta,
            life_rx_factor_alive,
            life_rx_factor_dead,
        }
    }
    const fn has_particles(&self) -> bool {
        self.has_live_particles() || self.has_dead_particles()
    }
    const fn has_live_particles(&self) -> bool {
        !self.particles[self.n_dead_particles].is_sentinel()
    }
    const fn has_dead_particles(&self) -> bool {
        !self.particles[0].is_sentinel()
    }
    const fn life_area_weight(&self, life: Life) -> float::T {
        match life {
            Life::Alive => self.live_area_weight,
            Life::Dead => self.dead_area_weight,
        }
    }

    pub fn burn(&self, terrain: &Terrain) -> Option<Fire> {
        if !self.has_particles() {
            None
        } else {
            let (rx_int, rbqig) = self.rx_int_rbqig(terrain);
            let speed0 = rx_int * self.flux_ratio / rbqig;
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
    pub fn burn_simple(&self, terrain: &Terrain) -> Option<FireSimple> {
        if !self.has_particles() {
            None
        } else {
            let (rx_int, rbqig) = self.rx_int_rbqig(terrain);
            let speed0 = rx_int * self.flux_ratio / rbqig;
            let (_phi_eff_wind, eff_wind, speed_max, azimuth_max) =
                self.calculate_wind_dependent_vars(terrain, speed0, rx_int);
            Some(FireSimple {
                speed_max: Velocity::new::<foot_per_minute>(speed_max),
                azimuth_max: Angle::new::<radian>(azimuth_max),
                eccentricity: Ratio::new::<ratio>(Self::eccentricity(eff_wind)),
            })
        }
    }
    fn rx_int_rbqig(&self, terrain: &Terrain) -> (float::T, float::T) {
        let mut wfmd = 0.0;
        let mut alive_moist = 0.0;
        let mut dead_moist = 0.0;
        let mut rbqig = 0.0;
        for p in self.particles.iter() {
            if p.is_sentinel() {
                break;
            }
            let m = p.moisture(terrain);
            rbqig += (250.0 + 1116.0 * m)
                * p.area_weight
                * self.life_area_weight(p.life)
                * p.sigma_factor;
            match p.life {
                Life::Alive => alive_moist += p.area_weight * m,
                Life::Dead => {
                    dead_moist += p.area_weight * m;
                    wfmd += m * p.sigma_factor * p.load
                }
            }
        }
        (
            self.life_rx_factor_alive * self.eta_m(Life::Alive, alive_moist, wfmd)
                + self.life_rx_factor_dead * self.eta_m(Life::Dead, dead_moist, wfmd),
            self.fuel_bed_bulk_dens * rbqig,
        )
    }
    #[inline]
    const fn eta_m(&self, life: Life, life_moist: float::T, wfmd: float::T) -> float::T {
        let life_mext = match (self.has_live_particles(), life) {
            (_, Life::Dead) => self.mext,
            (true, Life::Alive) => {
                let fdmois = wfmd * self.fine_dead_factor;
                let live_mext = self.live_ext_factor * (1.0 - fdmois * self.inv_mext) - 0.226;
                live_mext.max(self.mext)
            }
            (false, Life::Alive) => 0.0,
        };
        if life_moist >= life_mext {
            0.0
        } else {
            let rt = life_moist / life_mext;
            1.0 - 2.59 * rt + 5.11 * rt * rt - 3.52 * rt * rt * rt
        }
    }
    #[inline]
    fn calculate_wind_dependent_vars(
        &self,
        terrain: &Terrain,
        speed0: float::T,
        rx_int: float::T,
    ) -> (float::T, float::T, float::T, float::T) {
        let phi_ew = self.phi_ew(terrain);
        let speed_max1 = speed0 * (1.0 + phi_ew);
        let upslope = terrain.upslope();
        let wind_speed = terrain.wind_speed.get::<foot_per_minute>();
        let wind_az = terrain.wind_azimuth.get::<radian>();
        let ew_from_phi_ew = |p: float::T| (p * self.wind_e).powf(self.wind_b_inv);
        let max_wind = 0.9 * rx_int;
        let check_wind_limit = |pew: float::T, ew: float::T, s: float::T, a: float::T| {
            if ew > max_wind {
                let phi_ew_max_wind = self.wind_k * max_wind.powf(self.wind_b);
                let speed_max_wind = speed0 * (1.0 + phi_ew_max_wind);
                (phi_ew_max_wind, max_wind, speed_max_wind, a)
            } else {
                (pew, ew, s, a)
            }
        };
        use WindSlopeSituation::*;
        match Self::wind_slope_situation(terrain, speed0, phi_ew) {
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
                let x = slope_rate + wind_rate * split.cos();
                let y = wind_rate * split.sin();
                let rv = (x * x + y * y).sqrt();
                let speed_max2 = speed0 + rv;
                let phi_ew2 = speed_max2 / speed0 - 1.0;
                let eff_wind = ew_from_phi_ew(phi_ew2);
                let al = (y.abs() / rv).asin();
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
        terrain: &Terrain,
        speed0: float::T,
        phi_ew: float::T,
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
    fn hpua(&self, rx_int: float::T) -> float::T {
        rx_int * self.residence_time
    }
    fn eccentricity(eff_wind: float::T) -> float::T {
        let lw_ratio = 1.0 + 0.002840909 * eff_wind;
        (lw_ratio * lw_ratio - 1.0).sqrt() / lw_ratio
    }

    fn phi_slope(&self, terrain: &Terrain) -> float::T {
        let s = terrain.slope.get::<ratio>();
        self.slope_k * s * s
    }

    fn phi_wind(&self, terrain: &Terrain) -> float::T {
        let ws = terrain.wind_speed.get::<foot_per_minute>();
        self.wind_k * ws.powf(self.wind_b)
    }

    fn phi_ew(&self, terrain: &Terrain) -> float::T {
        self.phi_slope(terrain) + self.phi_wind(terrain)
    }
}

const fn safe_div(a: float::T, b: float::T) -> float::T {
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

const fn init_arr<T: Copy, const N: usize, const M: usize>(def: T, src: [T; M]) -> [T; N] {
    let mut dst = [def; N];
    let mut i = 0;
    while i < M {
        dst[i] = src[i];
        i += 1;
    }
    dst
}
#[allow(unused)]
pub(crate) fn fuzzy_cmp(msg: &str, a: float::T, b: float::T) -> bool {
    fuzzy_cmp_smidgen(msg, a, b, CMP_SMIDGEN)
}
#[allow(unused)]
pub(crate) fn fuzzy_cmp_azimuths(msg: &str, a: float::Angle, b: float::Angle) -> bool {
    let diff = (((a.get::<degree>() - b.get::<degree>()).abs() + PI) % 2.0 * PI - PI).abs();
    diff < 1.0
}
#[allow(unused)]
pub(crate) fn fuzzy_cmp_smidgen(msg: &str, a: float::T, b: float::T, smidgen: float::T) -> bool {
    let min = a.min(b).abs();
    let diff = (a - b).abs();
    let r = diff < smidgen || diff / min < MAX_FUZZY_CMP_DIFF;
    #[cfg(not(target_os = "cuda"))]
    #[cfg(test)]
    if !r {
        std::println!("{}: {} /= {}", msg, a, b);
    }
    r
}
