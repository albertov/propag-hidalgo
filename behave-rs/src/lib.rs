#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
extern crate uom;

pub mod behave;
pub mod types;
pub mod units;
