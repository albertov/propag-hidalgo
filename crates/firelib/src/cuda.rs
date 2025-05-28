use crate::float;
use crate::float::*;
use crate::from_quantity;
use crate::*;
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn cuda_standard_burn(
    model: &[u8],
    d1hr: &[float::T],
    d10hr: &[float::T],
    d100hr: &[float::T],
    herb: &[float::T],
    wood: &[float::T],
    wind_speed: &[float::T],
    wind_azimuth: &[float::T],
    slope: &[float::T],
    aspect: &[float::T],
    rx_int: *mut float::T,
    speed0: *mut float::T,
    hpua: *mut float::T,
    phi_eff_wind: *mut float::T,
    speed_max: *mut float::T,
    azimuth_max: *mut float::T,
    eccentricity: *mut float::T,
    residence_time: *mut float::T,
) {
    let i = thread::index_1d() as usize;
    if i < model.len() {
        let terrain = TerrainCuda {
            fuel_code: model[i],
            d1hr: d1hr[i],
            d10hr: d10hr[i],
            d100hr: d100hr[i],
            herb: herb[i],
            wood: wood[i],
            wind_speed: wind_speed[i],
            wind_azimuth: wind_azimuth[i],
            slope: slope[i],
            aspect: aspect[i],
        };
        if let Some(fire) = Catalog::STANDARD.burn(&terrain.into()) {
            let fire = Into::<FireCuda>::into(fire);
            *rx_int.add(i) = fire.rx_int;
            *speed0.add(i) = fire.speed0;
            *hpua.add(i) = fire.hpua;
            *phi_eff_wind.add(i) = fire.phi_eff_wind;
            *speed_max.add(i) = fire.speed_max;
            *azimuth_max.add(i) = fire.azimuth_max;
            *eccentricity.add(i) = fire.eccentricity;
            *residence_time.add(i) = fire.residence_time;
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn cuda_standard_simple_burn(
    len: usize,
    model: *const u8,
    d1hr: *const float::T,
    d10hr: *const float::T,
    d100hr: *const float::T,
    herb: *const float::T,
    wood: *const float::T,
    wind_speed: *const float::T,
    wind_azimuth: *const float::T,
    slope: *const float::T,
    aspect: *const float::T,
    speed_max: *mut float::T,
    azimuth_max: *mut float::T,
    eccentricity: *mut float::T,
) {
    let i = thread::index() as usize;
    if i < len {
        let terrain = TerrainCuda {
            fuel_code: *model.add(i),
            d1hr: *d1hr.add(i),
            d10hr: *d10hr.add(i),
            d100hr: *d100hr.add(i),
            herb: *herb.add(i),
            wood: *wood.add(i),
            wind_speed: *wind_speed.add(i),
            wind_azimuth: *wind_azimuth.add(i),
            slope: *slope.add(i),
            aspect: *aspect.add(i),
        };
        if let Some(fire) = Catalog::STANDARD.burn_simple(&terrain.into()) {
            *speed_max.add(i) = from_quantity!(Velocity, &fire.speed_max);
            *azimuth_max.add(i) = from_quantity!(Angle, &fire.azimuth_max);
            *eccentricity.add(i) = from_quantity!(Ratio, &fire.eccentricity);
        }
    }
}
