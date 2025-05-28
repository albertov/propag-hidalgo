#[cfg(feature = "std")]
extern crate std;

use crate::units::*;
use crate::units::areal_mass_density::pound_per_square_foot;
use crate::units::heat_flux_density::btu_sq_foot_min;
use crate::units::linear_power_density::btu_foot_sec;
use crate::units::radiant_exposure::btu_sq_foot;
use crate::units::reciprocal_length::reciprocal_foot;
use uom::si::angle::degree;
use uom::si::angle::radian;
use uom::si::available_energy::btu_per_pound;
use uom::si::f64::*;
use uom::si::length::foot;
use uom::si::mass_density::pound_per_cubic_foot;
use uom::si::ratio::ratio;
use uom::si::time::minute;
use uom::si::velocity::foot_per_minute;
use uom::si::velocity::meter_per_second;

use crate::types::*;

const SMIDGEN: f64 = 1e-6;
const PI: f64 = 3.141592653589793;


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

impl<'a> Spread {
    pub fn no_spread() -> Spread {
        Spread {
            rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(0.0),
            speed0: Velocity::new::<foot_per_minute>(0.0),
            hpua: RadiantExposure::new::<btu_sq_foot>(0.0),
            phi_eff_wind: Ratio::new::<ratio>(0.0),
            speed_max: Velocity::new::<foot_per_minute>(0.0),
            azimuth_max: Angle::new::<degree>(0.0),
            eccentricity: Ratio::new::<ratio>(0.0),
            residence_time: Time::new::<minute>(0.0),
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
        Length::new::<foot>(flame_length(self.byrams_max().get::<btu_foot_sec>()))
    }

    pub fn at_azimuth(&'a self, azimuth: Angle) -> Spreader<'a> {
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
            safe_div(1.0 - ecc, 1.0 - ecc * angle.cos())
        };
        Spreader {
            spread: self,
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

#[derive(Debug)]
pub struct Spreader<'a> {
    spread: &'a Spread,
    factor: Ratio,
}

impl<'a> Spreader<'a> {
    pub fn speed(&self) -> Velocity {
        self.spread.speed_max * self.factor
    }
    pub fn byrams(&self) -> LinearPowerDensity {
        self.spread.byrams_max() * self.factor
    }
    pub fn flame(&self) -> Length {
        Length::new::<foot>(flame_length(self.byrams().get::<btu_foot_sec>()))
    }
}

impl ParticleDef {
    const SENTINEL : Self = ParticleDef::standard(ParticleType::NoParticle, 0.0, 0.0);

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
    pub const fn is_sentinel(&self) -> bool {
        match self.type_ {
            ParticleType::NoParticle => true,
            _ => false
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
        let mut total = 0.0;
        let mut i = 0;
        while i < MAX_PARTICLES {
            let p = &particles[i];
            if !p.is_sentinel() {
                match (p.life(), self.life()) {
                    (Life::Alive, Life::Alive) | (Life::Dead, Life::Dead) => {
                        total += p.surface_area()
                    },
                    _ => ()
                }
                i += 1;
            } else {
                break
            }
        };
        safe_div(self.surface_area(), total)
    }
    const fn size_class_weight(&self, particles: &ParticleDefs) -> f64 {
        let mut total = 0.0;
        let mut i = 0;
        while i < MAX_PARTICLES {
            let p = &particles[i];
            if !p.is_sentinel() {
                match (p.life(), self.life()) {
                    (Life::Alive, Life::Alive) | (Life::Dead, Life::Dead) =>
                        match (p.size_class(), self.size_class()) {
                            (SizeClass::SC0, SizeClass::SC0) |
                            (SizeClass::SC1, SizeClass::SC1) |
                            (SizeClass::SC2, SizeClass::SC2) |
                            (SizeClass::SC3, SizeClass::SC3) |
                            (SizeClass::SC3, SizeClass::SC4) |
                            (SizeClass::SC5, SizeClass::SC5) => {
                        total += p.area_weight(particles)
                    },
                    _ => ()
                },
                _ => ()
                }
                i += 1;
            } else {
                break
            }
        };
        total
    }
    const fn surface_area(&self) -> f64 {
        let load = {
            let ArealMassDensity { value, ..} = self.load;
            value / (POUND_TO_KG / (FOOT_TO_M*FOOT_TO_M))
        };
        let savr = {
            let ReciprocalLength { value, ..} = self.savr;
            value / (1.0 / FOOT_TO_M)
        };
        let density = {
            let MassDensity { value, ..} = self.density;
            value / (POUND_TO_KG / (FOOT_TO_M*FOOT_TO_M*FOOT_TO_M))
        };
        safe_div(load * savr, density)
    }
    const fn sigma_factor(&self) -> f64 {
        let savr = {
            let ReciprocalLength { value, ..} = self.savr;
            value / (1.0 / FOOT_TO_M)
        };
        safe_div(-138.0, savr)
    }
}
const fn safe_div(a: f64, b: f64) -> f64 {
    if b > SMIDGEN {
        a / b
    } else {
        0.0
    }
}

impl Particle {
    const SENTINEL: Self = Particle::make(&ParticleDef::SENTINEL, &init_arr(ParticleDef::SENTINEL, []));
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
    pub const fn is_sentinel(&self) -> bool {
        match self.type_ {
            ParticleType::NoParticle => true,
            _ => false
        }
    }
    const fn moisture(&self, terrain: &Terrain) -> f64 {
        let Ratio { value, ..} = (match self.type_ {
            ParticleType::NoParticle => mk_ratio(9.0E100),
            ParticleType::Herb => terrain.herb,
            ParticleType::Wood => terrain.wood,
            ParticleType::Dead => match self.size_class {
                SizeClass::SC0 | SizeClass::SC1 => terrain.d1hr,
                SizeClass::SC2 | SizeClass::SC3 => terrain.d10hr,
                SizeClass::SC4 | SizeClass::SC5 => terrain.d100hr,
            },
        });
        value
    }
}
impl FuelDef {
    pub const fn standard<const N: usize, const M: usize, const F: usize>(
        name: [u8; N],
        desc: [u8; M],
        depth: f64,
        mext: f64,
        particles: [ParticleDef; F]
        ) -> Self {
        use std::marker::PhantomData;
        Self {
            name: init_arr(0,name),
            desc: init_arr(0, desc),
            depth: length_from_imperial(depth),
            mext: mk_ratio(mext),
            adjust: mk_ratio(1.0),
            particles: init_arr(ParticleDef::SENTINEL, particles)
        }
    }
}

impl Fuel {
    pub const SENTINEL : Self = Self::make(FuelDef::standard(*b"",*b"",0.0,0.0,[]));

    pub const fn make(def: FuelDef) -> Fuel {
        let mut alive_particles = [Particle::SENTINEL; MAX_PARTICLES];
        let mut dead_particles = [Particle::SENTINEL; MAX_PARTICLES];
        let mut i = 0;
        let mut i_a = 0;
        let mut i_d = 0;
        while i < MAX_PARTICLES {
            let p = &def.particles[i];
            if !p.is_sentinel() {
                    let p2 = Particle::make(p, &def.particles);
                    match p.life() {
                        Life::Alive =>  {
                            alive_particles[i_a] = p2;
                            i_a += 1;
                        },
                        Life::Dead =>{
                            dead_particles[i_d] = p2;
                            i_d += 1;
                        }
                    };
                    i += 1
                } else {
                    break
            };
        };
        let depth = length_to_imperial(&def.depth);
        let mext = extract_ratio(&def.mext);
        let adjust = extract_ratio(&def.adjust);
        Fuel {
            name: def.name,
            desc: def.desc,
            depth,
            mext,
            adjust,
            alive_particles,
            dead_particles,
        }
    }
    pub const fn has_particles(&self) -> bool {
        self.has_live_particles() || self.has_dead_particles()
    }
    const fn has_live_particles(&self) -> bool {
        let mut i = 0;
        while i>0 || i < self.alive_particles.len() {
           if self.alive_particles[i].is_sentinel() {
               break;
           }
           i += 1
        }
        i > 0
    }
    const fn has_dead_particles(&self) -> bool {
        let mut i = 0;
        while i>0 || i < self.dead_particles.len() {
           if self.dead_particles[i].is_sentinel() {
               break;
           }
           i += 1
        }
        i > 0
    }
    fn particles(&self) -> impl Iterator<Item = &Particle> {
        iter_particles(&self.dead_particles)
            .chain(iter_particles(&self.alive_particles))
    }
    const fn life_particles(&self, life: Life) -> &Particles {
        match life {
            Life::Alive => &self.alive_particles,
            Life::Dead => &self.dead_particles,
        }
    }
    fn total_area(&self) -> f64 {
        self.particles().map(|p| p.surface_area).sum()
    }
    fn life_area_weight(&self, life: Life) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.surface_area / self.total_area())
            .sum()
    }
    fn life_fine_load(&self, life: Life) -> f64 {
        match life {
            Life::Alive => iter_particles(self.life_particles(life))
                .map(|p| p.load * (-500.0 / p.savr).exp())
                .sum(),
            Life::Dead => iter_particles(self.life_particles(life))
                .map(|p| p.load * p.sigma_factor.exp())
                .sum(),
        }
    }
    fn life_load(&self, life: Life) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.size_class_weight * p.load * (1.0 - p.si_total))
            .sum()
    }
    fn life_savr(&self, life: Life) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.area_weight * p.savr)
            .sum()
    }
    fn life_heat(&self, life: Life) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.area_weight * p.heat)
            .sum()
    }
    fn life_seff(&self, life: Life) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.area_weight * p.si_effective)
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
    fn ratio(&self, sigma: f64, beta: f64) -> f64 {
        beta / (3.348 / sigma.powf(0.8189))
    }
    fn flux_ratio(&self, sigma: f64, beta: f64) -> f64 {
        ((0.792 + 0.681 * sigma.sqrt()) * (beta + 0.1)).exp() / (192.0 + 0.2595 * sigma)
    }
    fn beta(&self) -> f64 {
        safe_div(
            self.particles().map(|p| safe_div(p.load, p.density)).sum(),
            self.depth,
        )
    }
    fn gamma(&self, sigma: f64, beta: f64) -> f64 {
        let sigma15 = sigma.powf(1.5);
        let gamma_max = sigma15 / (495.0 + 0.0594 * sigma15);
        let aa = 133.0 / sigma.powf(0.7913);
        let rt = self.ratio(sigma, beta);
        gamma_max * rt.powf(aa) * (aa * (1.0 - rt)).exp()
    }

    fn life_rx_factor(&self, life: Life, sigma: f64, beta: f64) -> f64 {
        self.life_load(life)
            * self.life_heat(life)
            * self.life_eta_s(life)
            * self.gamma(sigma, beta)
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

    fn residence_time(&self, sigma: f64) -> f64 {
        384.0 / sigma
    }

    fn slope_k(&self, beta: f64) -> f64 {
        5.275 * beta.powf(-0.3)
    }

    fn wind_bke(&self, sigma: f64, beta: f64) -> (f64, f64, f64) {
        let wind_b = 0.02526 * sigma.powf(0.54);
        let r = self.ratio(sigma, beta);
        let c = 7.47 * ((-0.133) * (sigma.powf(0.55))).exp();
        let e = 0.715 * ((-0.000359) * sigma).exp();
        let wind_k = c * r.powf(-e);
        let wind_e = r.powf(e) / c;
        (wind_b, wind_k, wind_e)
    }
}

impl Combustion {
    pub fn make(fuel: Fuel) -> Self {
        let sigma = fuel.sigma();
        let beta = fuel.beta();
        let (wind_b, wind_k, wind_e) = fuel.wind_bke(sigma, beta);
        let life_rx_factor_alive = fuel.life_rx_factor(Life::Alive, sigma, beta);
        let life_rx_factor_dead = fuel.life_rx_factor(Life::Dead, sigma, beta);
        let total_area = fuel.total_area();
        Combustion {
            name: fuel.name,
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
            mext: fuel.mext,
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
    pub fn spread(&self, terrain: &Terrain) -> Spread {
        if !self.has_particles() {
            Spread::no_spread()
        } else {
            let rx_int = self.rx_int(terrain);
            let speed0 = self.speed0(terrain, rx_int);
            let (phi_eff_wind, eff_wind, speed_max, azimuth_max) =
                self.calculate_wind_dependent_vars(terrain, speed0, rx_int);
            Spread {
                rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(rx_int),
                speed0: Velocity::new::<foot_per_minute>(speed0),
                hpua: RadiantExposure::new::<btu_sq_foot>(self.hpua(rx_int)),
                phi_eff_wind: Ratio::new::<ratio>(phi_eff_wind),
                speed_max: Velocity::new::<foot_per_minute>(speed_max),
                azimuth_max: Angle::new::<radian>(azimuth_max),
                eccentricity: Ratio::new::<ratio>(Combustion::eccentricity(eff_wind)),
                residence_time: Time::new::<minute>(self.residence_time),
            }
        }
    }
    fn particles(&self) -> impl Iterator<Item = &Particle> {
        iter_particles(&self.dead_particles)
            .chain(iter_particles(&self.alive_particles))
    }
    const fn life_particles(&self, life: Life) -> &Particles {
        match life {
            Life::Alive => &self.alive_particles,
            Life::Dead => &self.dead_particles,
        }
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
        let ew_from_phi_ew = |p: f64| (p * self.wind_e).powf(1.0 / self.wind_b);
        let max_wind = 0.9 * rx_int;
        let check_wind_limit = |pew: f64, ew: f64, s: f64, a: f64| {
            if ew > max_wind {
                let phi_ew_max_wind = if max_wind < SMIDGEN {
                    0.0
                } else {
                    self.wind_k * max_wind.powf(self.wind_b)
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
                let x = slope_rate + wind_rate * split.cos();
                let y = wind_rate * split.sin();
                let rv = (x * x + y * y).sqrt();
                let speed_max2 = speed0 + rv;
                let phi_ew2 = speed_max2 / speed0 - 1.0;
                let eff_wind = {
                    if phi_ew2 > SMIDGEN {
                        ew_from_phi_ew(phi_ew2)
                    } else {
                        0.0
                    }
                };
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
                    * p.sigma_factor.exp()
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
            (lw_ratio * lw_ratio - 1.0).sqrt() / lw_ratio
        } else {
            0.0
        }
    }
    fn life_moisture(&self, life: Life, terrain: &Terrain) -> f64 {
        iter_particles(self.life_particles(life))
            .map(|p| p.area_weight * p.moisture(terrain))
            .sum()
    }

    pub const fn has_particles(&self) -> bool {
        self.has_live_particles() || self.has_dead_particles()
    }
    const fn has_live_particles(&self) -> bool {
        let mut i = 0;
        while i>0 || i < self.alive_particles.len() {
           if self.alive_particles[i].is_sentinel() {
               break;
           }
           i += 1
        }
        i > 0
    }
    const fn has_dead_particles(&self) -> bool {
        let mut i = 0;
        while i>0 || i < self.dead_particles.len() {
           if self.dead_particles[i].is_sentinel() {
               break;
           }
           i += 1
        }
        i > 0
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
            .map(|p| p.moisture(terrain) * p.sigma_factor.exp() * p.load)
            .sum()
    }
    const fn life_area_weight(&self, life: Life) -> f64 {
        match life {
            Life::Alive => self.live_area_weight,
            Life::Dead => self.dead_area_weight,
        }
    }

    fn phi_slope(&self, terrain: &Terrain) -> f64 {
        let s = terrain.slope.get::<ratio>();
        self.slope_k * s * s
    }

    fn phi_wind(&self, terrain: &Terrain) -> f64 {
        let ws = terrain.wind_speed.get::<foot_per_minute>();
        if ws > SMIDGEN {
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

#[derive(Debug)]
enum WindSlopeSituation {
    NoSpread,
    NoSlopeNoWind,
    WindNoSlope,
    SlopeNoWind,
    UpSlope,
    CrossSlope,
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use firelib_sys::*;
    use std::ffi::CString;
    use std::println;
    use std::vec::Vec;

    extern crate quickcheck;

    use quickcheck::{Arbitrary, Gen};

    quickcheck::quickcheck! {
        fn we_produce_the_same_output_as_firelib_c(terrain: Terrain, model: ValidModel, azimuth: ValidAzimuth) -> bool {
            let az = Angle::new::<degree>(azimuth.0);
            let (firelib_sp, firelib_sp_az) = firelib_rs_spread(model.0, &terrain, az);
            let (c_sp, c_sp_az) = firelib_c_spread(model.0, &terrain, az);
            Spread::almost_eq(&firelib_sp, &c_sp) && SpreadAtAzimuth::almost_eq(&firelib_sp_az, &c_sp_az)
        }
    }

    fn firelib_rs_spread(
        model: usize,
        terrain: &Terrain,
        azimuth: Angle,
    ) -> (Spread, SpreadAtAzimuth) {
        match STANDARD_CATALOG.get(model) {
            Some(fuel) if fuel.has_particles() => {
                let spread = fuel.spread(terrain);
                let spread_az = SpreadAtAzimuth::from_spreader(&spread.at_azimuth(azimuth));
                (spread, spread_az)
            }
            _ => (Spread::no_spread(), SpreadAtAzimuth::no_spread()),
        }
    }

    fn firelib_c_spread(
        model: usize,
        terrain: &Terrain,
        azimuth: Angle,
    ) -> (Spread, SpreadAtAzimuth) {
        unsafe {
            let name = CString::new("standard").unwrap();
            let catalog = Fire_FuelCatalogCreateStandard(name.as_ptr(), 13);
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
            Fire_SpreadAtAzimuth(
                catalog,
                model,
                (*fuel).azimuthMax,
                (FIRE_BYRAMS | FIRE_FLAME).try_into().unwrap(),
            );
            let f = *fuel;
            let spread = Spread {
                rx_int: HeatFluxDensity::new::<btu_sq_foot_min>(f.rxInt),
                speed0: Velocity::new::<foot_per_minute>(f.spread0),
                hpua: RadiantExposure::new::<btu_sq_foot>(f.hpua),
                phi_eff_wind: Ratio::new::<ratio>(f.phiEw),
                speed_max: Velocity::new::<foot_per_minute>(f.spreadMax),
                azimuth_max: Angle::new::<degree>(f.azimuthMax),
                eccentricity: Ratio::new::<ratio>(f.eccentricity),
                residence_time: Time::new::<minute>(f.taur),
            };
            Fire_SpreadAtAzimuth(
                catalog,
                model,
                azimuth.get::<degree>(),
                (FIRE_BYRAMS | FIRE_FLAME).try_into().unwrap(),
            );
            let f = *fuel;
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
            let models = &(0..13).collect::<Vec<_>>()[..];
            Self(*g.choose(models).unwrap())
        }
    }
    #[derive(Debug)]
    struct SpreadAtAzimuth {
        speed: Velocity,
        byrams: LinearPowerDensity,
        flame: Length,
    }

    impl SpreadAtAzimuth {
        fn from_spreader(s: &Spreader) -> Self {
            SpreadAtAzimuth {
                flame: s.flame(),
                speed: s.speed(),
                byrams: s.byrams(),
            }
        }
        fn no_spread() -> SpreadAtAzimuth {
            SpreadAtAzimuth {
                speed: Velocity::new::<foot_per_minute>(0.0),
                byrams: LinearPowerDensity::new::<btu_foot_sec>(0.0),
                flame: Length::new::<foot>(0.0),
            }
        }
        fn almost_eq(&self, other: &Self) -> bool {
            fn cmp(msg: &str, a: f64, b: f64) -> bool {
                let r = (a - b).abs() < SMIDGEN;
                if !r {
                    println!("{}: {} /= {}", msg, a, b);
                }
                r
            }
            cmp(
                "speed",
                self.speed.get::<foot_per_minute>(),
                other.speed.get::<foot_per_minute>(),
            ) && cmp(
                "byrams",
                self.byrams.get::<btu_foot_sec>(),
                other.byrams.get::<btu_foot_sec>(),
            ) && cmp("flame", self.flame.get::<foot>(), other.flame.get::<foot>())
        }
    }
}
