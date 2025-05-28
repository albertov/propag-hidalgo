use cuda_std::prelude::*;
use firelib_rs::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn standard_burn(model: &[usize], terrain: &[Terrain], result: *mut FireCuda) {
    let idx = thread::index_1d() as usize;
    if idx < terrain.len() {
        let elem = &mut *result.add(idx);
        if let Some(fire) = Catalog::STANDARD
            .get(model[idx])
            .and_then(|fuel| fuel.burn(&terrain[idx]))
        {
            *elem = fire.into()
        }
    }
}
