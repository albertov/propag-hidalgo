#![no_std]
#![feature(test)]

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
extern crate uom;

pub mod firelib;
pub mod types;
pub mod units;
