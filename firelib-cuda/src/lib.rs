#![feature(core_intrinsics)]
mod standard_burn;
pub use standard_burn::*;
mod propag;
pub use propag::*;
extern crate alloc;
