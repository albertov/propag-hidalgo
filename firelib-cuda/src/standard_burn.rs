use crate::propag::FireSimpleCuda;
use cuda_std::prelude::*;
use cuda_std::thread::sync_threads;
use firelib_rs::float;
use firelib_rs::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn standard_burn(
    model: &[usize],
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
        if let Some(fire) = Catalog::STANDARD.burn(model[i], &terrain.into()) {
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
pub unsafe fn standard_simple_burn(
    model: &[usize],
    d1hr: &[float::T],
    d10hr: &[float::T],
    d100hr: &[float::T],
    herb: &[float::T],
    wood: &[float::T],
    wind_speed: &[float::T],
    wind_azimuth: &[float::T],
    slope: &[float::T],
    aspect: &[float::T],
    speed_max: *mut float::T,
    azimuth_max: *mut float::T,
    eccentricity: *mut float::T,
) {
    let i = thread::index_1d() as usize;
    if i < model.len() {
        let terrain = TerrainCuda {
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
        if let Some(fire) = Catalog::STANDARD.burn_simple(model[i], &terrain.into()) {
            let fire = Into::<FireSimpleCuda>::into(fire);
            *speed_max.wrapping_add(i) = fire.speed_max;
            *azimuth_max.wrapping_add(i) = fire.azimuth_max;
            *eccentricity.wrapping_add(i) = fire.eccentricity;
        }
    }
}
