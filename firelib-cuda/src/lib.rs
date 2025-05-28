use cuda_std::prelude::*;
use firelib_rs::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn standard_burn(
    model: &[usize],
    d1hr: &[f64],
    d10hr: &[f64],
    d100hr: &[f64],
    herb: &[f64],
    wood: &[f64],
    wind_speed: &[f64],
    wind_azimuth: &[f64],
    slope: &[f64],
    aspect: &[f64],
    result: *mut FireCuda
) {
    let i = thread::index_1d() as usize;
    if i < model.len() {
        let elem = &mut *result.add(i);
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
        if let Some(fire) = Catalog::STANDARD
            .get(model[i])
            .and_then(|fuel| fuel.burn(&terrain.into()))
        {
            *elem = fire.into()
        }
    }
}
