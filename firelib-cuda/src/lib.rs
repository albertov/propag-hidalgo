#![feature(core_intrinsics)]

#[cfg(not(target_os = "cuda"))]
#[macro_use]
extern crate soa_derive;

#[macro_use]
extern crate firelib_rs;

mod standard_burn;
pub use standard_burn::*;
mod propag;
pub use propag::*;
//extern crate alloc;
